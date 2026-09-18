# 任务约束的关系修复：当前 TI2V 候选方法

方法标识：`task_grounded_relational_repair_v1`。主分支是 `combined_extended`，分离决策消融是 `factored_extended`。统一入口为 [method.py](method.py) 的 `propose`，生成与选择入口为 [run_ti2v_extended_generation.py](scripts/run_ti2v_extended_generation.py)。

## 方法及其针对的问题

机器人操作视频中的失败可能集中在动作关系上：物体尚未获得支撑就松手、接收手尚未抓稳就交接、推物时接触中断。本方法把原审查记录中的失败、任务文字和可见实体绑定到有限关系词表，再编译成一处技能文本修改。修改只作用于下一次生成；其有效性由生成后的固定五门槛和官方指标检验。

1. **沿用失败证据。** 读取父视频的原五门槛记录。没有明确 FAIL 时保留父技能；不将 UNKNOWN 改写为成功。输入绑定任务、首帧、父视频的 SHA-256。
2. **绑定实体并选择关系。** 两次固定提案调用读取首帧与均匀采样的视频帧。主分支联合记录实体、失败和关系，再做一次关系细化；消融先记录观察，再选择关系。第二次调用不能替换实体或扩大失败帧证据。
3. **检查关系是否适用于任务。** 沿用关系提案的 grounding 检查，并执行字面任务前提。可识别的任务族为拾取、放置、交接和推动；不能识别时弃权。例如，“把物体放到桌上”可以使用 `support_before_release`，但不能仅因模板存在就增加任务未要求的撤回动作。
4. **编译有限修改。** 从六种关系中选择一种，仅替换声明的技能槽位，单行至多 45 个英文单词。保留原任务、首帧、实体绑定和失败帧证据；拒绝的提案也保存。
5. **生成并按原规则选择。** 固定模型、相机约束、种子和生成参数。候选只有通过原五项门槛且修复父视频明确失败时才可替换父版本，否则回退。回退本身不表示父视频成功。

六种关系为：接触后运输、保持抓持、释放后撤回、支撑后释放、接收手抓稳后交接、推动中保持接触。完整模板见 [repair_factorial.py](integrations/skilladam_ti2v/repair_factorial.py)，编译约束见 [relational_repair.py](integrations/skilladam_ti2v/relational_repair.py)。任务前提是有限的英语规则，不是通用语言理解器。

## 相比 VideoWeaver，预期改进在哪里

[VideoWeaver: Evaluating and Evolving Skills for Agentic Long Video Generation](https://arxiv.org/abs/2606.08091) 已经使用执行轨迹、最终视频和中间证据评价并演化技能，不能把“使用反馈”或“演化技能”当成本方法独有的贡献。

这里要检验的差异是机器人 TI2V 的修正粒度：将修改限制为**任务确实需要、实体与失败帧支持的一条动作关系**。这可能减少附加动作和提示漂移，同时保留已正确的外观与场景。有限修改也允许在相同生成设置下隔离某一关系修正的影响。它是否改善 dyn、nDTW 等当前较弱的指标，需要视频结果验证；提案通过率无法回答这个问题。

| 组件 | 已观察到的作用 | 仍需验证的假设 |
|---|---|---|
| 扩展关系词表 | 主分支得到 17/60 条合格提案，覆盖 8 个案例 | 合格提案能改善生成后的动作关系 |
| 字面任务前提 | 拦住任务未要求的撤回动作；未知任务保留父技能 | 减少附加动作能提高任务一致性 |
| 单槽位有限修改 | 编译器限制修改位置和长度 | 改善动作的同时保留外观与相机 |
| 联合与分离决策 | 扩展词表下分别得到 17/60、13/60 条提案 | 联合决策的视频质量收益高于分离决策 |

## 与 VideoWeaver 的已完成对比

以下是**历史 Ours-v2**，不是本页新候选方法。统一评分采用 20 个开发案例、每例 3 个种子，共 60 条输出。VideoWeaver 行为 Qwen + MiniMax 的 TI2V 适配，历史优化成本并未匹配。

| 官方指标 ↑ | Ours-v2 | VideoWeaver 适配 | 配对差值 | 差值的 99% 区间 |
|---|---:|---:|---:|---:|
| BLEUScore | 0.215548 | 0.206252 | +0.009296 | [−0.049842, 0.067407] |
| CLIPScore | 89.223632 | 89.472307 | −0.248675 | [−1.902750, 1.424408] |
| hsd | 0.337533 | 0.332883 | +0.004650 | [−0.049184, 0.061953] |
| dyn | 0.250167 | 0.291683 | −0.041517 | [−0.116985, 0.029667] |
| ndtw | 0.333533 | 0.357150 | −0.023617 | [−0.102184, 0.046734] |

差值与区间原样取自服务器已完成的 [配对分析 JSON](evidence/historical-paired-analysis.json)：以案例为簇的配对 bootstrap，10,000 次，seed 20260918。五个区间均跨零；BLEU、hsd 的均值优势不能解释为整体显著优于 VideoWeaver。完整比较集合及负结果见 [RESULTS.md](RESULTS.md)。

## 当前候选的验证状态

运行 `20260918T095225Z` 比较未修改父版本、联合扩展主分支和分离扩展消融。三个分支都保留全部 60 条记录；完整输入与设置相同时共享相同提示的生成，共 77 条独立视频。目前已收到首条原生视频及哈希校验回执，整轮质量指标待评分。该运行使用自己的冻结实现和计划，本次整理出的统一入口不会替换运行中的源码。

本方法目前是未验证候选，尚不能作为“优于 VideoWeaver”或 SOTA 的结论。该开发集已多次用于方法选择；正式结论需要完整公开基准上的直接对比、组件消融和独立确认，不拼接不同版本各自最好的列。人工评审已取消，evaluator 改进由合作者负责。

## 调用接口

在授权服务器准备好已有 backend、父视频审查、技能和采样帧后：

```python
from pathlib import Path
from method import propose

plan = propose(
    backend, row, base_audit, parent_skill,
    Path(new_run) / "proposal", images, positions,
    arm="combined_extended",
)
# plan["skill"] 是下一次生成使用的技能；ABSTAINED 时保留父技能。
# plan.json 保存底层原始计划，method-decision.json 保存最终方法决策。
```

`row` 必须包含 instruction、initial、initial_sha256、base_sha256、seed。入口只转交这五个字段；官方分数、未来参考视频和竞争方法输出不作为提案输入。完整部署、依赖与生成协议见 [DEPLOYMENT.md](DEPLOYMENT.md) 和 [configs/paired-generation.example.json](configs/paired-generation.example.json)。
