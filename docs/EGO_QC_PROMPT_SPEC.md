# Ego 视频质检 Prompt 工程（0901 规范版）

本方案根据《灵生 Ego 数据质检规范-0901》整理，目标是让视觉模型同时产出：

1. 整条 episode 的 `valid` / `invalid` / `needs_human_review` 判断；
2. 可切除的 `bad` 时间片段；
3. 与平台审核原因对应的证据和时间戳。

模型不应接收数据集原有的 `audit_status`、`invalid_reason` 或带有标签含义的 `bad` 文本，否则会造成标签泄漏。

## 1. 输入字段

每次调用给模型一条 episode，建议传入：

```json
{
  "episode_id": "dataset-9049/episode-000000",
  "task_category": "李学臣桌面植物养护，放种子",
  "task_instruction": "将准备好的种子放到培育盒子里喷水并盖上盖子",
  "duration_sec": 66.0,
  "camera_view": "egocentric_stereo_left_rgb",
  "movement_context": "stationary | walking | unknown"
}
```

任务指令清洗规则：

- 优先使用人工维护的任务指令表；
- 从 `meta/episodes.jsonl` 或 `meta/tasks.jsonl` 中排除 `bad...`、`record action with object spatial relation`、URL、单字母和乱码；
- 如果只能得到任务类别，明确告诉模型任务细节不足，并允许输出 `needs_human_review`；
- 不要把 `bad` 文本当作审核结论。

视频输入优先使用 `left_rgb`，`right_rgb` 作为遮挡或视野不清时的复核视角。初轮不使用深度视频，避免把深度编码问题混入质检结果。

## 2. 判定层级

### 2.1 整条视频判定

`valid` 的必要条件：

- 任务主体和关键步骤基本完成；
- 画面确实是第一人称采集视角；
- 手部、道具和关键动作可观察；
- 没有隐私、人脸、无关设备屏幕或明显干扰；
- 没有持续的静止、无意义动作、冻结、黑屏、严重模糊或丢帧。

以下任一情况清晰成立时，整条视频判为 `invalid`：

- 非第一人称视角，拍摄他人操作；
- 任务没有实际意义、只是测试片段，或任务未完成；
- 中途出现其他人脸、地址、手机号等隐私信息；
- 无关手机/平板/电脑屏幕出现并被触碰或使用；
- 超过 10 秒静止、黑屏、无效图像、休息、聊天、玩手机、抓痒、摸鼻子等无关动作；
- 抽烟、喝水、吃东西等采集外行为；
- 重复动作、刻意演示、明显抖腿或头部大幅晃动；
- 手部长期不可见、距离镜头过近、手指频繁出画面，导致关键动作无法判断；
- 道具长期不在视野、画面被头发/手/物体挡住、严重过暗/过亮或模糊；
- 视频频繁闪烁、冻结、黑屏、缺少动作片段或无法连续播放；
- 任务过程中的关键视线、姿态或操作明显不符合采集要求，例如中途抬头/回头回应他人、勾手腕或手臂贴桌导致关节姿态异常；
- 摄像头固定角落阴影、中心区域模糊、过热模糊或明显冷暖色切换，已经影响任务判断；
- 画面上下颠倒不作为当前规则单独判定项（规范原文已删除该条）。

### 2.2 可切除的 `bad` 片段

下面的情况优先标记时间片段为 `bad`，不要自动把剩余有效任务整条判为无效：

- 静止任务开头超过 2 秒没有手部出现；单手出现不算问题；
- 视频开头约 10 秒是无关或无意义动作；
- 一个视频包含多个任务时，将较短的任务片段标为 `bad`，保留持续时间最长且符合任务的片段；
- 视频开头出现其他人脸，但后续片段没有人脸且任务完整；
- 只影响局部、切除后不影响主任务判断的短暂遮挡或无关动作。

如果 `bad` 片段切除后没有剩余有效任务，则整条 episode 仍判为 `invalid`。

### 2.3 走动任务的例外

先判断任务是否需要走动，再判断手部是否出画：

- 手里拿着采集物品时，走动途中手和道具可以短暂离开视野；如果手或道具重新出现在画面中，应尽量同时出现；
- 手里没有采集物品时，正常走动可以短暂离开视野；
- 不能把走动任务的短暂出画，误判为“手部露出少”。

## 3. 错误标签与平台原因映射

模型输出细粒度 `error_tags`，并另外输出一个平台可选的 `platform_invalid_reason`。

| 细粒度标签 | 平台原因建议 |
| --- | --- |
| `task_not_completed` | `未按要求完成任务` |
| `extra_action`、`repeated_or_demo` | `包含多余动作` |
| `no_motion_over_10s` | `全程静止无动作` |
| `device_screen` | `采集设备屏幕露出` |
| `unexpected_interference`、`face`、`privacy` | `发生意外干扰` 或 `其他` |
| `hand_not_visible`、`hand_too_close` | `手部露出少` |
| `blur`、`center_blur`、`heat_blur`、`occlusion`、`too_dark_or_bright`、`freeze`、`missing_segment`、`corner_shadow`、`color_shift` | `视频模糊不清` 或 `其他` |
| `non_ego_view`、`look_away`、`wrist_hook`、`hand_too_close`、`object_out_of_view` | `其他` 或 `手部露出少` |
| 无法归入以上类别 | `其他`，并在 `notes` 中说明具体原因 |

平台原因只有一个字段时，选择造成审核不通过的主要原因；其他观察到的问题放入 `secondary_reasons`，不要丢失证据。

## 4. 推荐 Prompt

### System Prompt

```text
你是严格的第一人称 Ego 视频质检员。你的任务是根据任务要求和视频证据，判断数据是否可保留，并标出需要切除的 bad 时间片段。

只依据视频中实际看到的内容，不得猜测没有出现的动作，不得读取或推断任何隐藏的人工审核标签。先检查整条视频，再检查时间片段。必须区分“整条无效”和“有效视频中的局部 bad 片段”。

任务完成是首要标准；视频质量、第一人称视角、隐私、人脸、无关设备屏幕、手部/道具可见性和采集员无关行为是质量门禁。

如果证据不足，输出 needs_human_review，不要强行猜测。输出必须是合法 JSON，不要输出 Markdown、解释性前缀或额外文本。
```

### User Prompt 模板

```text
请审核下面这条 Ego 视频。

episode_id: {episode_id}
任务类别: {task_category}
任务要求: {task_instruction}
视频时长（秒）: {duration_sec}
是否需要走动: {movement_context}

请按以下顺序检查：
1. 是否为第一人称视角；
2. 任务关键步骤和最终状态是否完成；
3. 手部、道具和关键区域是否可观察；
4. 是否出现人脸、隐私、手机/平板/电脑屏幕；
5. 是否有超过 10 秒的静止、无关或无意义动作；
6. 是否有重复演示、抽烟、喝水、吃东西、聊天、抬头回应他人等干扰；
7. 是否有闪烁、冻结、黑屏、丢帧、遮挡、模糊、过暗/过亮或明显色彩变化；
8. 对开头无手、开头无关动作、开头人脸和多任务视频，判断是否可以只切除局部 bad 片段。

只输出下面结构的 JSON：
{
  "episode_review": "valid | invalid | needs_human_review",
  "task_completion": "complete | partial | not_completed | unclear",
  "platform_invalid_reason": null,
  "error_tags": [],
  "secondary_reasons": [],
  "segments": [
    {
      "start_sec": 0.0,
      "end_sec": 0.0,
      "action": "keep | bad",
      "reason_code": "task_not_completed | no_motion_over_10s | device_screen | face | privacy | extra_action | blur | center_blur | heat_blur | occlusion | freeze | missing_segment | corner_shadow | color_shift | hand_not_visible | hand_too_close | object_out_of_view | look_away | wrist_hook | other",
      "evidence": "只描述视频中可回看的事实"
    }
  ],
  "evidence": [
    {
      "start_sec": 0.0,
      "end_sec": 0.0,
      "observation": "可复核的观察"
    }
  ],
  "confidence": 0.0,
  "needs_human_review": false,
  "notes": "简短说明"
}

要求：
- valid 时 platform_invalid_reason 必须为 null；
- invalid 时 platform_invalid_reason 必须是：未按要求完成任务、其他、包含多余动作、采集设备屏幕露出、全程静止无动作、发生意外干扰、手部露出少、视频模糊不清 之一；
- 如果整条视频无效，segments 应覆盖主要无效区间并标为 bad；
- 如果只有局部问题且剩余任务有效，episode_review 可以为 valid，并把问题区间标为 bad；
- 每个时间戳必须在 0 到视频时长之间；
- 走动任务不能因为手部短暂出画就直接判无效；
- 不要把任务文本中的 bad 当作审核结论。
```

## 5. 视频下采样、切片与上传

不要把几十到几百 MB 的原始 1080p 视频直接转成 Base64 放进 JSON。Base64 会进一步放大请求体，而且长视频会让模型把注意力分散到大量无关帧。建议保留原始视频不变，另建临时预处理文件：

### 5.1 两阶段采样

**粗扫阶段**（低成本定位）：

- 360p 或 480p；
- 2 FPS；
- 保留原始时间轴；
- 重点找人脸、手机/平板屏幕、黑屏/冻结、长静止、画面遮挡和任务切换。

**复核阶段**（用于最终判定）：

- 720p；
- 6～8 FPS，手部快速动作不建议低于 6 FPS；
- 只复核粗扫命中的时间段、任务开始/结束段和前后 5～10 秒；
- 复核窗口之间保留 2～3 秒重叠。

不能只按 10 秒或 30 秒抽一帧，因为短暂人脸、屏幕、手部出画和丢帧都可能被漏掉。

### 5.2 长视频切片

- 60 秒以内：通常可以整段复核；
- 60～180 秒：按 45～60 秒切片，片段之间重叠 2～3 秒；
- 超过 180 秒：先粗扫，再只对风险窗口做高质量复核；
- 每个片段都必须携带 `source_start_sec`、`source_end_sec`，模型输出的相对时间要加回原始偏移。

示例（只生成临时文件，不覆盖原视频）：

```bash
# 粗扫：360p、2 FPS，适合快速定位风险区间
ffmpeg -i input.mp4 -vf "scale=-2:360,fps=2" \
  -c:v libx264 -preset fast -crf 30 -an coarse.mp4

# 复核：720p、8 FPS，保留低码率音频用于判断聊天/干扰
ffmpeg -ss 120 -t 60 -i input.mp4 \
  -vf "scale=-2:720,fps=8" \
  -c:v libx264 -preset medium -crf 27 \
  -c:a aac -b:a 64k fine-0120.mp4
```

### 5.3 上传方式

优先级如下：

1. 方舟 File API 或可访问的 HTTPS/TOS URL；
2. 已压缩、已切片且体积较小的临时 MP4；
3. 当前 `test_llm_api.py` 的本地文件 Base64，仅用于小片段试验，不用于原始长视频。

上传清单建议记录：

```json
{
  "source_video": ".../file-000.mp4",
  "segment_file": ".../fine-0120.mp4",
  "source_start_sec": 120.0,
  "source_end_sec": 180.0,
  "fps": 8,
  "width": 1280,
  "height": 720,
  "sha256": "..."
}
```

这样模型只看到预处理片段，但最终证据仍然可以准确回到原视频时间轴。

## 6. 推荐执行流程

1. 从 manifest 建立任务清洗表，不将 `audit_status` 和 `invalid_reason` 传给模型；
2. 对视频做粗扫，得到候选风险区间；
3. 用 720p、6～8 FPS 复核候选区间及任务开始/结束段；
4. 每条 episode 先使用 `left_rgb`，视野遮挡或证据冲突时再复核 `right_rgb`；
5. 输出 `segments` 和 JSON 结果，并将相对时间换算为原视频时间；
6. 将模型结果与 manifest 离线对比，统计二分类和原因分类指标；
7. 人工复核所有低置信度样本，并将错误案例回写为 Prompt 的反例。

## 7. 首轮 10 条评估指标

首轮建议使用 5 条 `valid` 和 5 条 `invalid`，并覆盖任务未完成、无关动作、设备屏幕、静止等原因。模型结果与 manifest 的真实标签分离保存，至少统计：

- `episode_review` 的准确率、invalid 召回率和 F1；
- `platform_invalid_reason` 的准确率和宏平均 F1；
- `needs_human_review` 占比；
- `segments` 的时间区间是否落在实际问题附近；
- 是否发生“任务完成但被误判无效”或“明显错误却判有效”。

初轮只测单次调用，不使用真实标签重试，避免 Prompt 在 10 条样本上过拟合。
