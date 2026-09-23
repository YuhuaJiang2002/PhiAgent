# 基于 RVE 改造 Ego 数据二元质检 —— 设计与 Prompt 方案

> 基础：`/data1/review/rve`（Robot Video Evaluation v3.3.4，22 个原子 / 5 维度 / LangGraph 多 Agent）
> 目标：把"评生成视频的任务完成度"改造成"**判真实采集数据 keep / delete 的二元质检器**"
> 重点：**大模型针对性提问的重设计**（用户指定的核心工作）
> 状态：设计稿，未动手

---

## 一、先看清 rve 有什么（已读源码）

### 1.1 流水线阶段（`graph_orchestrator.py` 实测）

```
prepare_evidence → event_locator → target_localizer → geometry_observer
   → [identity_tracker ‖ entity_ledger] → identity_skeptic → reconcile_identity → identity_adjudicator
   → dimension_fanout（M0/M1/M2/M3/M6 并行）
   → plan_frame_refill → refill_evidence → refill_rejudge（原子级补帧闭环）
   → evidence_validator → deterministic_scoring → END
```

### 1.2 原子契约结构（23 个原子，7 个 critical）

```python
{"atom_id": "destination_achievement", "weight": 0.35, "critical": True, "optional": False,
 "min_pass_frames": 5, "min_pass_span_s": 0.75,
 "pass_rule": "...", "partial_rule": "...", "fail_rule": "..."}
```

### 1.3 七个**可以直接继承**的设计（这是 rve 最值钱的部分）

| # | 机制 | 为什么对质检关键 |
|---|---|---|
| 1 | **模型不决定最终结论**："score_0_to_4 和 verdict 只是模型自检值，确定性代码会按原子契约权重重新计算并覆盖它们" | 质检必须可审计、可复现，结论不能由 LLM 自由裁量 |
| 2 | **hard_gate 由代码强制**："critical atom credit=0 会由代码自动建立 hard gate，模型无法用 hard_gate=false 抑制" | 天然适合"人脸出现即一票否决" |
| 3 | **观察与判定分离**：`geometry_observer_prompt` 注释直言 "Ask a visual role for observations, **never for a task verdict**" | 避免模型在观察阶段就被引导下结论 |
| 4 | **证据强制**：每条判定必须给 `frame_ids`；"合法帧号只证明引用存在，**不证明帧内容支持主张**" | 直击 VLM 幻觉 |
| 5 | **反证与不确定性**：必须同时输出 `evidence` / `counter_evidence`；`uncertain` 不计分 | 质检里"证据不足"必须能表达 |
| 6 | **Reverse Audit + Adjudicator**：主动找反例推翻已有结论，并有严格规则防止退化成橡皮图章 | 对抗"模型一律判通过"的退化 |
| 7 | **盲测隔离**：不得从文件名/标签推断，上游 Agent 文字只是**待核实假设**不是真值 | 质检不能用人工标签当输入（否则评测无意义） |

### 1.4 必须**扔掉**的部分

- **5 个维度的语义**：M0 参考一致性 / M1 关键目标 / M2 动作序列 / M3 功能接触 / M6 可判定性 —— 这是"机器人任务做对了吗"，不是"采集合不合规"
- **0–100 分制与 QUALITY_WEIGHTS**：质检要二元，分数无意义
- **身份追踪链**（identity_tracker / entity_ledger / identity_skeptic / reconcile / adjudicator）：为"是否操作了正确对象"服务，质检里只有"道具不在视野"需要一点，可大幅精简

---

## 二、改造后的目标架构

### 2.1 新流水线（在 rve 骨架上增删）

```
prepare_evidence（增强：兼顾均匀采样 + 连续窗口）
   → segment_locator        【替代 event_locator】定位任务段/异常段/边界
   → signal_observer        【新增】把 IMU/时间戳/手部等**数值信号**转成可引用事实
   → visual_observer        【替代 geometry_observer】只报可见事实，不下结论
   → rule_fanout            【替代 dimension_fanout】Q1/Q2/Q3/Q4 四组并行
   → plan_frame_refill → refill_evidence → refill_rejudge   【保留】
   → evidence_validator     【保留】
   → binary_decision        【替代 deterministic_scoring】输出 keep / delete / needs_human
```

### 2.2 四组质检规则（替代 M0/M1/M2/M3/M6）

| 组 | 内容 | 判据性质 | 典型处理 |
|---|---|---|---|
| **Q1 隐私与合规** | 人脸露出、隐私信息（地址/手机号）、手机平板、非第一人称 | 语义+检测 | **一票否决** |
| **Q2 画面可用性** | 光线过暗/过亮/剧变、模糊、中心区模糊、四角阴影、画面颠倒、闪烁冻结黑屏、丢帧、抖动 | **可计量** | 阈值 + 证据 |
| **Q3 任务与动作** | 未按要求完成任务、全程静止、多余/重复动作、道具不在视野、手部露出少、勾手腕、手距过近 | 信号+语义 | 加权 |
| **Q4 可判定性** | 证据是否足以支撑以上判断（继承 M6 的作用） | 元判断 | 决定 needs_human |

### 2.3 二元判定的确定性聚合（替代评分）

```python
def decide(atoms, coverage):
    # 1) 硬否决：任一 VETO 原子 fail → delete（代码强制，模型无法覆盖）
    if any(a.credit == 0 and a.critical for a in atoms if a.group == "Q1"):
        return "delete", "veto:" + a.atom_id
    # 2) 画面不可用：Q2 关键原子 fail 达阈值 → delete
    if quality_defect_score(atoms) >= Q2_DELETE_THRESHOLD:
        return "delete", "quality"
    # 3) 任务类：Q3 加权失败 → delete
    if weighted_fail(atoms, "Q3") >= Q3_DELETE_THRESHOLD:
        return "delete", "task"
    # 4) 证据不足或覆盖度不够 → 交人工，不猜
    if coverage.insufficient or any(a.unknown for a in critical_atoms):
        return "needs_human", "insufficient_evidence"
    return "keep"
```

**关键**：阈值全部是**代码里的常量**，可调、可审计、可按业务偏好（怕误删 vs 怕漏放）整体平移。

---

## 三、Prompt 重设计（本次的核心）

### 3.1 系统提示：从"评任务"改成"判可用性"

**现有（rve）**
```
你是机器人生成视频的视觉证据审核员。只报告图片中直接可见的事实，不读取或猜测数据集故障标签。
…任务描述中的"成功完成"也是规范目标，不是候选视频已经成功的事实。
```

**建议（QC）**
```
你是第一人称（Ego）采集数据的质检证据审核员。你的判断将决定一条数据被保留还是删除。

【铁律】
1. 只报告画面中直接可见的事实。不得从文件名、目录名、数据集名或任务名推断任何结论。
2. 你的上游 Agent 输出只是"待核实假设"，不是事实。文字与画面冲突时以画面为准，并显式标记冲突。
3. 任何 fail 判定必须给出可直接核对的帧号；帧号合法不等于帧内容支持你的主张，你必须亲自核对画面语义。
4. "没有看到问题"不能作为 pass 的证据。只有当你要求的帧数与时间跨度都满足、且画面确实清晰可判，
   才能判 pass；否则必须判 unknown。
5. 判 unknown 不会被惩罚，编造证据会被追溯。证据不足时一律 unknown。
6. 你只回答被问到的那一条规则，不评价其他规则，不做总体结论。总体结论由确定性代码生成。
7. 只输出一个合法 JSON 对象，不要输出 Markdown。
```

> 与 rve 的差异：新增第 4 条（反"没看见即通过"）、第 6 条（禁止越界下总论），第 5 条明确 unknown 免责 —— 这三条是针对质检场景的**防退化**设计。

### 3.2 原子契约：把规范 27 条翻译成可判定的三档规则

这是 prompt 的"判据来源"。原则：**每条规则必须写成"在什么时间窗内、看几个帧、看到什么算 pass/partial/fail"**。

**示例：规范 2.12 光线过暗或过亮 → Q2 原子**

```python
{
  "atom_id": "illumination_extreme",
  "group": "Q2", "critical": False, "weight": 0.15,
  "min_pass_frames": 4, "min_pass_span_fraction": 0.30,
  "measurement": {"metric": "subjective_hand_visibility", "unit": "enum"},
  "pass_rule": "覆盖约 30% 时长的至少 4 个帧中，执行任务的手部轮廓与手指分界清晰可辨（能说出在抓什么/在哪）。",
  "partial_rule": "手部可见但细节吃力：能看出手的整体位置与动作方向，但看不清手指与所持对象的边界；发生在部分时段。",
  "fail_rule": "连续时段内手部区域过暗或过曝成剪影/白斑，无法判断手在做什么；或全程如此。",
}
```

**示例：规范 2.5 人脸露出 → Q1 一票否决原子**

```python
{
  "atom_id": "human_face_visible",
  "group": "Q1", "critical": True, "weight": 1.0, "min_pass_frames": 1,
  "pass_rule": "在全部已采样帧中均未出现人脸（含镜面反射、照片、屏幕中的人脸）。",
  "partial_rule": "存在疑似人脸但分辨率不足无法确认（必须给出帧号，交人工）。",
  "fail_rule": "任一处出现可辨认的人脸（他人或本人，含镜面/照片/屏幕）。",
}
```
> 注意 `pass_rule` 写"全部已采样帧"是刻意的：这会把"采样覆盖不足"暴露成 Q4 的问题，而不是让模型假装看全了。

**示例：规范 2.21 丢帧 → 由 `signal_observer` 提供数值，模型只做解释**

```python
{
  "atom_id": "timestamp_gap",
  "group": "Q2", "critical": False, "weight": 0.20,
  "measurement": {"metric": "max_frame_interval_ratio", "unit": "x", "basis": "timestamp 序列"},
  "pass_rule": "最大帧间隔不超过标称间隔的 2 倍，且无超过 0.3s 的空洞。",
  "partial_rule": "存在 2–5 倍间隔的空洞，但时间占比 <5%，且不跨关键动作。",
  "fail_rule": "存在 >5 倍间隔的空洞，或空洞跨越关键动作导致动作不连贯。",
}
```
> 这条**不该让 VLM 判**——数值直接来自 parquet 的 `timestamp`。模型只负责"空洞是否跨越关键动作"这一半。

### 3.3 针对性提问：把开放问题改成封闭问题（**最关键**）

这是用户点出的核心。VLM 对"这条视频有没有问题"这种开放问题会给出敷衍答案，必须**逐条封闭提问**。

**反面（要避免）**
```
请判断这条视频是否存在质量问题。输出 PASS / FAIL。
```

**正面（建议）**

```
角色：Q2-illumination Agent。

【本次要判的唯一一条规则】
规范 2.12：光线过暗或过亮，看不清手部动作算无效；能看清手部动作算有效。

【输入】
- 图像顺序（引用帧必须用这里的 ID）：
  f001(0.00s) f002(1.20s) ... f024(28.4s)
- 覆盖度声明：本视频 292.3s，采样 24 帧（均匀 8 帧 + 细节 16 帧，细节集中在 0–45s 与 250–292s）

【必须逐项回答（只答这一条规则）】
1. 在手部出现的时段内，手部区域是否可辨认？（不可辨认 / 部分可辨认 / 可辨认 / 无法判断）
2. 若判"不可辨认"或"部分可辨认"：给出至少 4 个帧 ID 与对应时间，说明看不的是什么
   （手部整体轮廓？手指与所持对象的边界？），并说明该时段占视频的比例。
3. 反证：有没有哪些帧显示手部其实是清楚的？给出帧 ID。
4. 若你没有覆盖到手部出现的时段，必须回答"无法判断"并说明原因。

【输出】
严格按此结构（其余字段保持 null/空）：
{"skill":"Q2","criterion":"illumination", "verdict":"pass|partial|fail|unknown",
 "score_0_to_4":null, "atomic_checks":[{"atom_id":"illumination_extreme",
   "status":"pass|partial|fail|unknown","credit":null,"frame_ids":[],
   "reason":"","measurement":{"metric":"subjective_hand_visibility","value":null,"unit":"enum","basis":""}}],
 "confidence":0.0,"hard_gate":false,
 "evidence":[{"frame_ids":[],"claim":""}],"counter_evidence":[{"frame_ids":[],"claim":""}],
 "judgeability":"sufficient|partial|insufficient","need_more_evidence":false}
```

**这套问法的七个要点**（建议固化成模板约束）：

| # | 要点 | 作用 |
|---|---|---|
| 1 | **一次只问一条规则** | 避免规则间相互冲淡，也让失败可归因 |
| 2 | **给封闭选项**（不可辨认/部分/可辨认/无法判断） | 不许自由发挥 |
| 3 | **要求可核对的帧 ID + 时间** | 幻觉无处藏 |
| 4 | **强制反证** | 抵消"一律判失败"的模型偏见 |
| 5 | **明示覆盖度**，并要求覆盖不足时答"无法判断" | 防"没看到就算通过" |
| 6 | **给出该规则的原文** | 让模型按规范判，而非按自己的审美 |
| 7 | **明确"只答这一条，不做总论"** | 把结论权收回给确定性代码 |

### 3.4 技术性指标的锚定（VLM 的短板，必须补）

VLM 没有"抖动多厉害算抖"的绝对标尺。三种补法，建议组合：

- **(a) 数值转文本**：由 `signal_observer` 把 IMU 陀螺高频能量、帧间光流残差等算成数字，写进 prompt 作为**已知事实**，模型只解释。
- **(b) 比较式提问**：不问"抖不抖"，而问"比较 f003 与 f018，画面内容的位置偏移是否明显大于 f003 与 f005 之间的偏移"。
- **(c) 少样本锚定**：在 prompt 里附 2–3 张已标注的代表帧（轻度/中度/重度），让模型对齐尺度。

### 3.5 任务卡的替换

- rve 的 `task_card`（任务 SOP）**保留** —— 因为"未按要求完成任务"这条要用它
- 新增 `rule_card`（质检规则卡）= 规范条目的可观察化表述，作为 Q1/Q2 的判据来源
- `applicability_terms` 机制沿用：例如"光线"类原子只在……其实 QC 里 Q2 全时段适用，Q1 全时段适用；Q3 的"道具不在视野"才需要按任务卡激活

---

## 四、改造点清单（文件级）

| 文件 | 改动 |
|---|---|
| `robot_eval/prompts.py` | **重写**：BASE_SYSTEM、删除 DIMENSION_* → 新增 QC_GROUP_*；重写 `dimension_prompt` → `rule_prompt`；保留 `evidence_validator_prompt` / `reverse_audit_prompt` / `audit_adjudicator_prompt` 骨架并改写措辞 |
| `robot_eval/atomic_contracts.py` | **重写** `ATOM_SPECS`：27 条规范 → Q1/Q2/Q3/Q4 原子，标 critical/weight/min_pass_frames |
| `robot_eval/outcome_scoring.py` | **替换**为 `binary_decision.py`：硬否决 + 加权阈值 + needs_human |
| `robot_eval/scoring.py` | 保留分数计算工具，但不再用于最终结论 |
| `robot_eval/graph_orchestrator.py` | 增 `signal_observer`；`event_locator`→`segment_locator`；精简身份链；`dimension_fanout`→`rule_fanout` |
| `config/` | 新增 `qc_rule_cards_v1.json`（27 条可观察化规则）+ `qc_eval_v1.json` |
| **新增 `detectors/`** | 纯 CPU 信号层（无需 LLM，见第五节） |
| `docs/` | 新增 `QC-FRAMEWORK-v1.md` 记录契约与阈值依据 |

---

## 五、两阶段落地：先无 LLM，再有 LLM

**阶段 A（零 API 成本，先出基线）**
纯 CPU 信号层，直接判 Q2 的技术类子集 + Q3 的部分：IMU 抖动、静止时长、丢帧、亮度/色温、清晰度分区、手部出画（用现成的 `keep_frame`/`left_kept`）、人脸检测。**在 8195 条标注上测召回/精确率**，得到"纯规则能覆盖多少"的真实数字（我预估按原因约 12–30%，见 v2 方案）。

**阶段 B（复用 rve 骨架 + 新 prompt）**
1. 先用 **300–500 条分层抽样**跑通全链路（控制成本），逐条对照人工标签，重点看：模型的 fail 判定里有多少能通过 `evidence_validator`（即证据是否真的支持）
2. 按规则分组统计召回 → 找出"模型判不了"的规则，回到 3.4 用数值/比较/少样本补
3. 达标后再全量跑

---

## 六、必须提前定的技术风险

1. **帧采样策略冲突**：rve 均匀采样 32 帧是为了看"任务流程"；质检的**抖动/闪烁/冻结/丢帧必须先看连续帧**才能发现。需要"均匀采样 + 异常区段加密采样"的双通道，且 `prepare_evidence` 得改。
2. **VLM 对技术指标不可靠**：必须靠阶段 A 的数值兜底，不能指望模型看出丢帧。
3. **成本与延迟量级**：按已有 VLM 链路的实测（10 条 episode / 约 1400s / 约 1.19 元，4 并发），全量 8195 条约需 **数十小时与近千元**量级（并发提高可缩短）。建议先用分层样本评估，不要全量盲跑。
4. **"其他"占 34%**：人工标注里最大的一类是兜底"其他"，其内涵未知。**二元任务不受影响**（照样是 delete），但会影响我们对"哪条规则重要"的判断 —— 建议抽样一批"其他"人工看一眼，摸清里面到底是什么。
5. **规范无量化阈值**：所有阈值需从 8195 条标注反推或先给经验值再校准，契约里必须记录阈值来源与版本。

---

## 七、需要你确认

1. **保留多少 rve 的身份追踪链？** 我建议砍掉（QC 用不上），但"道具不在视野"可能会用到目标定位 —— 可以只保留 `target_localizer`。
2. **Q4 可判定性 是否独立成组**（继承 M6）？我建议保留，它直接决定"交人工"的流量。
3. **错误偏好**：误删 vs 漏放，哪个更不能接受？（决定所有阈值的整体平移方向）
4. **是否先做阶段 A**（纯规则基线，零成本当天出数）？我建议先做，它能立刻告诉你"LLM 到底需要补多少"。
