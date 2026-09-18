# 运行环境与复现接口

## 源码和依赖

本目录的纯逻辑模块不需要 CUDA、PyTorch 或模型权重。模型调用由 `TI2VBackend` 的服务适配器完成，SkillAdam 只在注册或运行其优化器时导入。

远端控制端需要 FFmpeg/FFprobe、`jsonschema`，以及能运行 XGrammar 的独立 Python 环境。`protocol.json` 中的 `grammar_python` 指向该环境。生成服务、Qwen 服务和官方评分服务使用各自已经验证的运行环境，不能把控制端依赖当成它们的完整安装清单。

生成服务需要 MiniMax-H3 对应分区的完整权重证明、固定 runtime 源码和已确认的 GPU。服务端必须保存设备选择、CUDA_VISIBLE_DEVICES、UUID、进程归属、命令、软件版本及原生视频回执。发布包不复制第三方模型仓库。

## 一次配对生成需要的文件

| 文件/目录 | 要求 |
|---|---|
| `source/` | 本目录的固定源码副本；从该目录导入 |
| `protocol.json` | `expected_hostname`、期限、原生/辅助调用预算、`generator`、`auxiliary`、`grammar_python`、`native_pools` |
| `experiment.json` | `reserved_at`、`max_seconds`、`generation_workers`、`owned_native_pools`、`stage_budgets.generate` |
| `inputs.json` | `records`，每条包含 `case_id`、`seed`、原始 `instruction`、`initial`、`initial_sha256`、`base`、`base_sha256` |
| `parent-selections.json` | 同一完整清单下的 `base_audit`，不改写历史 FAIL/UNKNOWN |
| `planning/plans-frozen.json` | `arms` 为 parent/combined_extended/factored_extended；每条记录含完整三臂 `prompts`；`readiness=READY_FOR_GENERATION` |
| `planning/state.json` | 上述计划文件的 `plans_sha256` |
| `source-manifest.json` | 冻结输入、配置和源码的相对路径 → SHA-256；不包含会变化的输出 |
| `qwen.lock` | 所有共用 Qwen 服务的任务共享同一个锁目标 |
| `score-queue/` | 官方评分请求与结果队列，由独立评分服务处理 |

`auxiliary` 至少提供实际服务的 `endpoint` 和 `served_model`，另保存精确模型 revision、服务进程和 GPU 绑定。CPU 控制进程要求 `CUDA_VISIBLE_DEVICES=''`，GPU 服务进程必须使用验证后的物理设备。示例清单见 `configs/paired-generation.example.json`；占位符必须在服务器上解析并冻结。

当前开发清单检查固定为 20 个案例及三个种子 20260910、20260911、20260912。更换数据集时必须新建协议，并同时更新清单验证，不能把开发结果直接当成完整测试集成绩。

## 服务契约

原生生成池在 `queue/` 接受 `.job.json`，返回对应 `.job.result.json`，包括路径、SHA-256 和生成回执。池的 `execution.json` 必须报告可用状态。模型初始化的 warmup 也计入真实成本；池调用上限需要覆盖 warmup 和所有不同提示。

评分请求的 `selected` 列表含案例、种子、视频路径与哈希，只有选择清单冻结后才提交。评分结果必须为 `SCORED`，包含逐案例的五项指标及官方证据哈希。部署现有 EWMBench wrapper 时应固定原始版本、逐视频字幕种子和视频处理协议。评分服务不可向提案器返回未来参考或在选择前泄漏官方分数。

历史队列部署与跨服务器传输由项目环境提供；本发布包重点收录方法、控制器和验证代码。没有服务时应报告等待，不能在工作站启动模型、基准评测或实验。

## 结果解释

同模型复核、语法通过和模板许可不是人工真值，也不是视频质量结果。原始候选与选择结果均完整报告，并按案例聚类保留三个种子，避免把 60 条视频当成 60 个独立场景。详细环境与逐调用证据留在实验归档，公开摘要只保留所需统计和绑定哈希。
