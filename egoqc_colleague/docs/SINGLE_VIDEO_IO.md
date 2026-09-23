# 单条视频评测：输入与输出

入口：`tools/evaluate_one.py`。复用 `qc_run_v3.py` 的 v3db 提示词、s2 抽帧和 `decide()` 判定，原脚本与历史结果不修改。
这是单条调用演示，尚未实现或启动 FastAPI 服务。

## 业务输入

只允许三个非空字符串字段：`request_id`、`video_path`、`task_description`。多余字段，包括人工标签和人工分段，会被拒绝。
公开输入模板见 `examples/single.input.example.json`，需自行填写已获准使用的视频路径和完整 SOP。
真实输入、视频和推理产物不随源码发布。历史单条实测使用 v3db 流程与当时输入清单的 SOP，不以此重算历史指标。

```json
{
  "request_id": "demo-single-video",
  "video_path": "/absolute/path/to/left_video.mp4",
  "task_description": "将散落在桌面或收纳区的积木、玩具模块按颜色或形状分类，逐一放入对应的收纳箱中，确保每个模块归位整齐"
}
```

本机将视频抽为 H/A/T/E/B/C 六组图片。模型只接收图片、分组标题、估计时间标签、质检提示词和 SOP。
不向模型发送视频路径、request_id、dataset_id、人工标签、无效原因、人工分段或 Parquet。

## 本地准备（不联网）

```bash
cd egoqc_colleague
python tools/evaluate_one.py \
  --input examples/single.input.example.json \
  --env-file .env \
  --model YOUR_ENABLED_MODEL_ID \
  --output runs/NEW_PREVIEW_DIRECTORY \
  --prepare-only
```

输出目录必须不存在，防止覆盖。`prepared_not_submitted` 只表示本地准备完成，不是模型推理成功。

## 实际评测

须先获准将视频抽帧与 SOP 发到方舟接口，之后去掉 `--prepare-only` 并使用新的输出目录。
从 `.env.example` 创建本地 `.env` 并显式传入 `--env-file .env`，覆盖原开发环境默认路径；使用 `ARK_API_KEY` 和 `ARK_BASE`／`ARK_BASE_URL`。
用 `--model` 指定账号已开通的模型 ID，`--think auto`。密钥只用于请求认证，不保存到结果或日志。
仅发一次主请求，不做原版六次自动重试，也不开启不影响最终判定的额外反证审计。
鉴权、额度或网络错误写入 `error.json`，不会伪造 valid/invalid 结果。

## 文件

| 文件 | 内容 |
|---|---|
| `input.json` | 业务输入，未来可映射为 API 请求体 |
| `frames/` | 实际抽帧 JPEG |
| `sampling.json` | 视频信息、帧标签、估计时间、图片 SHA256 |
| `model_request.json` | 模型请求体，含图片 Base64，不含认证头 |
| `model_request.preview.json` | 人眼可读预览；图片 Base64 替换为路径提示，不能原样发送 |
| `prepared.json` | 只在本地准备模式生成，明确未联网 |
| `model_response.json` | 成功调用后才生成的回复正文、返回模型名、token、耗时等 |
| `model_output.txt` | 模型完整输出正文，不截断为旧版的 1500 字符 |
| `output.json` | 成功取得模型回复后生成的结构化业务返回 |
| `error.json` | 调用失败信息，不是视频判定结果 |

## 返回字段约定（并非模拟推理结果）

- `request_id` / `status`：请求标识和处理状态。
- `verdict`：`keep`、`delete` 或 `needs_human`。
- `review_status`：对应 `valid`、`invalid` 或 `needs_review`。
- `triggered_rule`：最终命中的代码规则，如 `device_or_suspect`、`task_incomplete`。
- `checks`：模型五项检查、证据帧、覆盖情况及置信度。
- `validation_errors`：输出 JSON、必填字段、枚举、证据帧编号检查结果；不合格则转人工。
- `crop`：当前固定为 `{ "supported": false, "start_sec": null, "end_sec": null }`。
- `pipeline`：提示词／抽帧版本、请求与返回模型名、提示词和源码 SHA256。
- `video` / `frame_count` / `usage` / `timing` / `warnings`：视频信息、帧数、消耗、耗时和限制。

`delete` 是质检建议，不会删除任何源文件。

## 封装 FastAPI 前的边界

1. 人工答案应留在服务外的评测层，不能送进推理函数。
2. 公网接口应用视频 ID 或受控存储路径，不应允许任意读取服务器文件。当前 CLI 不是安全的公网服务实现。
3. 抽帧和推理适合后台任务，接口返回任务 ID，再查询状态和结果；需要限制并发、视频大小和时长。
4. 密钥留在服务端，不放业务请求体，不把图片 Base64 写进访问日志。
5. 这是旧版首尾固定 2 秒豁免策略，不是新项目的动态裁剪规则。没有可靠的裁剪时间输出，不能硬填“2 秒到时长减 2 秒”冒充识别结果。
6. s2 时间标签是抽帧规则估计值，不是解码器真实帧 PTS；未来精确裁剪需要另行核验。

单条试跑只验证调用链和输入输出，不证明整套测试集效果或生产可用性。
