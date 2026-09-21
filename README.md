# Ego Video QC

一个独立、无 GPU 依赖的 ego 视频基础质检工具。这个项目位于
`ego-video-qc` 孤儿分支，分支历史不继承 PhiAgent `main`。

## 能检查什么

工具通过 `ffprobe` 和 `ffmpeg` 检查：

- 视频是否能读取，以及是否包含视频流；
- 最小分辨率、帧率和时长；
- 采样帧是否大面积全黑/全白；
- 连续采样帧是否长期冻结，以及视频是否完全静止。

它不会自动声称视频一定是第一人称 ego 视角，也不判断手部、物体接触或任务是否完成；这些项目需要人工或模型复核。

## 使用

环境需要 `ffprobe` 和 `ffmpeg`，Python 仅使用标准库：

```bash
python -m ego_video_qc path/to/ego.mp4
python -m ego_video_qc path/to/ego.mp4 --json --output qc-report.json
```

退出码：`0` 表示自动门禁通过，`1` 表示视频可读但至少一项门禁失败，`2` 表示工具或输入错误。

可调参数示例：

```bash
python -m ego_video_qc clip.mp4 \
  --min-width 640 --min-height 360 --min-fps 15 \
  --min-duration 2 --max-duration 120 --sample-fps 4
```

## LLM 视频质检 Prompt

根据《灵生 Ego 数据质检规范-0901》整理的整条审核、局部 `bad` 切片、错误标签和结构化 JSON 输出方案见：
[`docs/EGO_QC_PROMPT_SPEC.md`](docs/EGO_QC_PROMPT_SPEC.md)。

150 条批量测试的误差分析和改进建议见：
[`docs/PILOT_150_ERROR_ANALYSIS.md`](docs/PILOT_150_ERROR_ANALYSIS.md)。

## 测试

```bash
python -m pytest
```

测试不需要 GPU、模型权重或真实视频文件。
