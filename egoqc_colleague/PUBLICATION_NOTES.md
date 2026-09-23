# 源码导入说明

同事项目以 `egoqc_colleague/` 子目录加入 `YuhuaJiang2002/PhiAgent` 的新分支 `egoqc_qy`。
新分支基于原有 `ego-video-qc` 分支；保留其原有文件和历史，不修改远端旧分支，也不导入同事项目原来的 `.git`。

## 包含

- 原有 Python 脚本、提示词版本，不修改判定规则。
- 除含凭据交接文件外的 Markdown 设计、迭代和测评文档。
- 单条调用入口、通用占位输入、环境配置示例和依赖清单。
- 原 README 改名为 `README_LEGACY.md`；新 README 区分历史脚本、单条演示和未实现的 API 能力。

## 不包含

- `HANDOVER.md`：原文包含凭据，不得上传。
- `.env`、密钥、令牌、SSH 登录信息及凭据文件。
- `frames/`、`runs/`、`logs/`、`out/` 和 Python 缓存。
- 视频、数据集、Parquet、真实输入清单、逐条 `preds_*.jsonl`、`labels_corrected_v1.tsv`、`review_52_human.json`。
- `metrics_*.json` 等运行产物；汇总指标见历史 Markdown 报告。
- `docs/质检规范_全文.txt`：内部规范原文不随公开源码发布。
- 同事项目旧 Git 历史和标签，避免带入历史敏感文件。

以上排除只影响 GitHub 上传副本，不删除或修改原工作目录文件。
历史报告中的数据路径、样本编号和结果引用可能指向未公开的本地文件；这不是完整数据发布。

## 后续开发

模型参数、SOP 和数据路径须在本地配置。单条入口显式传入 `--env-file .env` 和已开通的模型 ID。
FastAPI 服务、URI 映射、LDP 查询、持久化队列和幂等仍待实现；没有为了适配接口而改变推理结论。
