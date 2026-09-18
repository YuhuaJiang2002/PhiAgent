# TI2V：任务约束的关系修复

本目录提供当前 TI2V 候选方法：将失败视频中的任务、实体和动作关系绑定后，编译成一处有限技能修改，再用固定的生成与选择协议验证。具体机制、相对 VideoWeaver 的改进假设和完整对比见 [METHOD.md](METHOD.md)，统一调用入口为 [method.py](method.py)。

当前候选的视频质量评分尚未完成。历史 Ours-v2 在 BLEU、hsd 均值上领先 VideoWeaver 适配版，其余三项落后；不能据此宣称整体更好。模型推理和官方评分只在授权服务器运行；纯合成单元测试可在普通 CPU 上运行。

当前主线固定原五项视频门槛、选择规则和 EWMBench 官方评分，改进生成提示及技能。人工评审已取消，evaluator 改进由合作者负责，本目录没有启用新的双向偏好评审器。

## 代码入口

| 路径 | 内容 |
|---|---|
| `method.py` | 当前候选的统一提案入口、任务前提和决策留存 |
| `integrations/skilladam_ti2v/backend.py` | 远端生成、原五门槛审查、选择和官方评分队列 |
| `integrations/skilladam_ti2v/adapter.py` | 可选的官方 SkillAdam 适配器 |
| `integrations/skilladam_ti2v/relational_repair.py` | 证据绑定、关系模板、有限文本替换 |
| `integrations/skilladam_ti2v/repair_factorial.py` | 联合/分离决策 × 原始/扩展词表的四臂提案实验 |
| `phiagent/harness/video_meta_rsi.py`、`video_pareto_rsi.py` | 有界技能与策略优化、保留/回退规则 |
| `scripts/run_ti2v_extended_generation.py` | 冻结提示后的三臂配对生成：父版本、联合扩展、分离扩展 |
| `scripts/verify_ti2v_repair_factorial.py` | 原始返回、图像字节、预算与决策重放检查 |
| `tests/` | 不读取真实数据、不调用模型的合成检查 |

这是从工作仓库整理出的独立源码快照。`SOURCE_MANIFEST.json` 记录文件哈希，远端各次实验另有自己的冻结源码清单。运行时从本目录启动，避免与仓库根目录中的同名模块混用。SkillAdam 上游源码、模型权重、数据和机器凭据不随本目录发布。

## 快速检查

需要 Python 3.10 或更高版本；这些测试只使用标准库，可选 SkillAdam 合约测试在上游未安装时跳过。

```bash
cd experiments/ti2v
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## 实验顺序

1. 在服务器准备完整的案例/种子清单，记录任务、首帧、基础视频及其哈希。当前开发集为 20 个案例，每例 3 个固定种子，所有分支保留全部 60 条记录。
2. 固定模型版本、采样、预算、源码和 Git 状态，确认生成服务的物理 GPU 与 UUID。控制进程不直接使用 GPU。
3. 运行提案实验并复核其原始请求。保留失败、弃权和无修改记录；合格提案数只衡量提案覆盖率。
4. 在新的实验目录冻结生成计划。所有分支共用模型、首帧、时长、分辨率和种子。相同输入及完整生成设置下，字节相同的提示可共享生成结果。
5. 全部视频生成并完成选择后，锁定选择清单，再运行官方评分。分别报告原始候选和最终选中输出。

服务器入口示例：

```bash
# 在授权服务器上，RUN 指向新建并完成配置冻结的实验目录。
# RUN/source 是本目录源码副本；不得对旧实验目录重新执行。
CUDA_VISIBLE_DEVICES='' python3 scripts/run_ti2v_extended_generation.py \
  --root "$RUN" --stage generate
```

实验目录、依赖、服务接口与输入格式见 [DEPLOYMENT.md](DEPLOYMENT.md)。这些入口保留原运行环境的主机验证；未经配置不能直接在另一台机器上运行。这里没有提供一条命令自动部署模型的安装器。

## 当前证据

[RESULTS.md](RESULTS.md) 保留开发集对比、未奏效的修正，以及最新提案检查。扩展词表的配对视频实验正在推进，质量结果待评分。现有结果尚不支持 TI2V SOTA。

本轮生成比较使用联合扩展为主候选、未修改父版本为对照、分离扩展为消融。原始词表的部分修改引入未被任务要求的撤回动作，因此没有进入本轮生成；其提案记录保留在诊断中。这个选择发生在视频质量评分前，仍属于已使用开发集上的自适应实验。

## 模型与基准版本

见 [configs/model-pins.json](configs/model-pins.json)。生成后端仅限 MiniMax/JoyAI，本轮继承 MiniMax-H3；没有 JoyAI 对比结果。五项官方指标为 BLEUScore、CLIPScore、hsd、dyn、ndtw。公开完整基准、匹配的强基线和独立确认是后续工作。
