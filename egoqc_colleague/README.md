# Ego 视频质检：同事项目源码快照

本目录保存同事项目的质检脚本、提示词和测评文档，与仓库根目录原有的 `ego_video_qc/` 项目分开维护。
本次导入没有修改质检规则，也没有把单条调用演示包装成已上线的 API 服务。

## 入口与版本

| 路径 | 用途 |
| --- | --- |
| `tools/evaluate_one.py` | 单视频调用演示；复用 `qc_run_v3.py` 的 v3db 规则，只发一次主模型请求 |
| `qc_run_v3.py` | 108 条数据评测使用的 v3db 流程 |
| `qc_run_v26.py` | 已保存的 v26 迭代脚本；不是单条演示默认使用的版本 |
| `qc_run*.py`、`prompts/` | 不同历史实验的脚本和提示词 |
| `docs/SINGLE_VIDEO_IO.md` | 单条输入、输出与能力边界 |
| `v3全量108条测评结果_20260922.md` | 历史测评报告；指标对应当时的数据与版本 |
| `README_LEGACY.md`、`HISTORY.md` | 原始项目说明和历史记录，保留当时口径 |
| `PUBLICATION_NOTES.md` | 导入范围与未上传的内容 |

## 单条调用

需要 Python 3.11、`ffmpeg` 和 `ffprobe`，以及已获准使用的视频、任务 SOP 和模型账号。

```bash
cd egoqc_colleague
python -m pip install -r requirements.txt
cp .env.example .env
```

在本地 `.env` 中填写密钥，不要提交到 Git。将输入模板中的占位路径和 SOP 替换为获准使用的真实输入。
人工有效/无效标签和人工标注片段不得作为模型输入。

先只准备抽帧和请求预览，不调用模型：

```bash
python tools/evaluate_one.py \
  --input examples/single.input.example.json \
  --env-file .env \
  --model YOUR_ENABLED_MODEL_ID \
  --output runs/local-preview \
  --prepare-only
```

实际调用需获准将视频抽帧和 SOP 发送给配置的服务商，然后去掉 `--prepare-only` 并换一个不存在的输出目录。
`--model` 必须使用账号已开通的模型 ID；历史报告不构成模型可用性承诺。
显式使用 `--env-file .env`，覆盖原开发机默认配置路径。

## 当前边界

- 尚未提供 FastAPI 的 `submit` / `poll`、任务队列或幂等服务。
- 单条入口输出 `keep`、`delete` 或 `needs_human`；`delete` 是建议，不会删除源视频。
- 沿用首尾固定 2 秒豁免策略，不支持可靠的精确裁剪时间；抽帧时间是估计值。
- 旧批量脚本保留原环境的绝对路径，运行前须检查数据根目录、输出目录、索引文件和模型配置；v23–v26 支持 `QC_DATASET_ROOT`。
- 历史数据路径和测评记录不表示数据已随本仓库发布，也不代表可以直接复现实验。

密钥、原始数据、抽帧、逐条推理结果和旧 Git 历史未导入。原项目完整副本保留在原工作目录。
