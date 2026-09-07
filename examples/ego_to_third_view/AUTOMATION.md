# 批量 ego → 第三人称控制 → DiT → 视觉审查/返工

## 能力边界（先读）

本次把 v21 后实际有效的修正整理为可复用代码，并保留 v35/v7/v8 的单视频证据。
新的批量执行器不包含物体名、长边方向、桌面尺寸、12秒动作节点或固定 GPU 编号。
它支持不同已重建场景的统一数据接口、批量状态管理、几何检查、生成前后审查、限次返工和交付。

**不能据此声称已经做到“任意原始 RGB 视频，无标定、无人审查，必定正确生成”。**
当前仍需外部感知适配器或已审查的世界坐标 bundle，提供手/物体/尺度/操作者位置。
旧 HaWoR/SAM2/VGGT/SAM3D/FoundationPose 脚本有场景专用输入，不能冒充通用自动感知。
缺少尺度或操作者侧别证据时返回 `NEEDS_INPUT`。当前通用人体 rig 支持站位基本不变的操作；
走动、多人、复杂多物体共抓、长于15秒的连续 H3 生成需要独立适配与验证，不会偷偷套桌面模板。

CPU 测试覆盖不同物体名、四个操作者朝向、世界平移、相机移动、两视频批处理/恢复/返工隔离。
这些是接口与几何回归，不是多真实场景视觉质量验收。新增通用 mesh renderer 和 resident worker
保留对应运行证据等级；不要把旧 v35/v7 实测自动算成新代码已经端到端跑通。

## 硬约束与场景参数

| 永远保留 | 每个视频重新确定 |
| --- | --- |
| 动作从原 ego 操作者在世界中的位置发出，不从第三人称观察相机发出 | 操作者在桌子长边/短边/其他位置，root、heading、证据和置信度 |
| 上臂/前臂长度按人物确定后保持恒定；不可达就修尺度/根位置/轨迹，不能拉长手臂 | 合理人体测量值、肩宽；不是所有人固定0.36/0.32m |
| 两只手臂连接同一个稳定躯干，保留平滑 yaw，抑制整个人随手抖动 | 身体朝向、视野裁剪、yaw 限幅；走动需专门模型 |
| 保留手物对应、动作顺序、末尾仍持有的状态和遮挡关系 | 物体 ID/形状/材质、抓持区间、可观测对称性 |
| EGO 为时间依据，SIM 同步；DiT 不能擅自加速或截断参考尾部 | 视频时长、源 PTS/CFR 导出映射、事件节点 |
| 相机与 actor 根位置解耦；不以换镜头掩盖人体问题 | 新场景经审查的固定第三人称机位 |
| 加速需真实执行证据与原门槛校验；不跨 NUMA、不抢占其他任务 | GPU UUID 组、空闲显存、软件版本、外部模型路径 |

“白板背景、绿色垫子、闹钟背面朝观察者、蓝/灰圆柱”属于此例的场景事实，
不是跨视频强加的规则。通用要求是保持**该视频自己的**环境与物体朝向。
几何是估计值；粗体网格/贴合约束不等于力闭合、无碰撞或真实物理仿真认证。

## 执行路径

```text
每个原 ego + 感知适配器/已审查 bundle
  → reconstruct → stabilize → render
  → 全帧视觉审查（不通过则上游返工；不会先花钱生成 DiT）
  → 补齐 H3 参考尾帧 → 常驻 H3 worker（串行多请求、独立缓存 epoch）
  → 全帧视觉审查
      ├─ 几何/接触问题：回到对应上游阶段
      ├─ DiT 画面问题：新候选重生成，保留旧候选
      └─ 只有时序问题：平滑分段重映射，仅选取原 DiT 帧
  → 再审查 → EGO | SIM | DiT 三联交付
```

每个 clip 的状态/尝试目录独立，默认最多2次返工。一个视频失败不删除、不覆盖，也不阻止其他 clip 处理。
`NEEDS_REVIEW` / `NEEDS_INPUT` / `WAITING_GPU` / `REPAIR_BUDGET_EXHAUSTED` 均不等于完成。
恢复时核对 manifest、源文件与中间产物哈希；换输入、修改参数或手工覆盖中间视频需新 batch，旧审查失效。
网格目录下的网格、材质和图像文件也绑定哈希；请将场景资产放在独立目录，避免扫描无关大目录。
纹理应放在同一网格目录或其子目录中，不使用指向目录外的材质依赖。

```bash
# 只检查/显示计划，不启动模型。
python examples/ego_to_third_view/batch.py \
  --manifest /data/my-batch.json --output /data/runs/batch-001

# 明确执行；适配器在指定外部环境运行。
python examples/ego_to_third_view/batch.py \
  --manifest /data/my-batch.json --output /data/runs/batch-001 --execute

# 填完 pending_review 的 review.json 后继续，不重跑已完成阶段。
python examples/ego_to_third_view/batch.py \
  --manifest /data/my-batch.json --output /data/runs/batch-001 --execute --resume
```

从 [batch.example.json](batch.example.json) 填入自己的路径。示例中的两个视频是**路径模板，不是已经通过的跨场景样本**。
有待审查时退出码2，完成全部 clip 才返回0。查看 `summary.json` 与各 clip 的 `state.json`、`history.jsonl`。
没有填写 `review_pre`/`review_post` 适配器时默认等待人工或当前 Codex 审查，绝不伪造自动 VLM 通过。

## 世界坐标 bundle 接口

```text
bundle/
  scene.json       # 场景和人体参数、source_sha256、物体/静态资产列表
  events.json      # fps/frames/source_sha256、事件与接触区间
  motion.npz       # timestamps_s、手网格和物体 world_from_object
  ego.mp4          # 审查过的源时钟导出（H3 当前路径为24fps）
  meshes/...       # 相对 scene.json/bundle 的 OBJ/MTL/纹理，不上传受限资产
```

`scene.json` 核心结构（数字仅演示格式，必须从自己的源证据确定）：

```json
{
  "schema_version": 1,
  "source_sha256": "实际原ego文件的64位SHA256",
  "coordinate_system": {"frame":"scene_world","unit":"m","up":"+z","handedness":"right"},
  "actor": {
    "origin_authority":"source_ego_world",
    "origin_evidence":"源相机标定、手臂朝向和支撑面的侧别证据路径/说明",
    "origin_confidence":0.9,
    "root_m":[0,-0.5,0.3], "heading_rad":0,
    "root_motion":"stationary", "length_policy":"constant_per_actor",
    "upper_arm_m":0.34, "forearm_m":0.29, "shoulder_width_m":0.36
  },
  "render_camera":{"eye_m":[0.8,-0.8,0.6],"target_m":[0,0,0.1],"width":1024,"height":768,"fov_y_deg":55},
  "objects":[{"id":"mug-7","mesh":"meshes/mug.obj","description":"ceramic mug","orientation_mode":"object"}],
  "static_meshes":[{"path":"meshes/scene.obj"}],
  "appearance_description":"此场景自身的背景和材质约束"
}
```

Heading 0 对应局部 +y；root 为肩部参考高度。相机 eye/target 只用于观察，不能据此反推 actor。
若纹理弱对称物体的轴向旋转不可观测，可显式 `orientation_mode: world_locked` 并提供 `symmetry_evidence`；
默认完整跟随物体旋转，不能按物体名字猜“圆柱所以锁死”。

NPZ 以 `allow_pickle=False` 读取：`timestamps_s[T]`、`left_hand_world_m[T,V,3]` / `right_hand_world_m`、
相应 `left_faces[F,3]` / `right_faces`、每个任意对象 ID 的 `ID__world_from_object[T,4,4]`。
单手可用，但未观测的手不能无证据新增。当前手网格要求开放腕部边界；闭合网格需上游导出明确腕部索引适配。
`world_from_object` 使用列向量齐次矩阵：`p_world = p_object @ R.T + t`。不能直接混入 HaWoR 相机坐标。

`events.json` 的 `events` 元素包含 `id,time_s,confidence,evidence`，可附 `object_id,action`；
`contacts` 元素包含 `object_id,side,start_frame,end_frame,confidence,evidence`，闭区间，不能同手重叠绑定两个物体。
`automation.event_detection.propose_events` 可从校准物体速度与观测接触区间提出开始/停止/松手候选；
候选需在生成前审查。结束时仍在抓持，不会凭空新增松手。源 VFR 应先保存 PTS→CFR 对照，
然后同一时钟驱动 EGO、轨迹和 SIM，不允许每阶段自行重采样。

## 感知与审查适配器

`adapters.reconstruct/stabilize/render` 可替换为本地可信命令数组：

```json
{"argv":["/env/bin/python","/project/my_frontend.py","--context","{context}","--output","{output}"],"gpu_count":1,"min_free_mib":12000,"timeout_s":1800}
```

context 包含 source、source_sha256、clip、已有 artifacts、settings、硬约束和 repair_history。
适配器在新 output 目录写 `result.json`，返回绝对产物路径。reconstruct 至少返回 scene/motion/events/ego；
stabilize 增加 rig/geometry_report；render 再增加 sim。执行器重算 NPZ 臂长、共享肩膀、根位置与 yaw-only 检查，
不只相信 report 的布尔值。不同 Python/CUDA 环境可逐阶段选择，不全局安装模型。

视觉适配器格式：

```json
{"argv":["/env/bin/python","/project/my_visual_reviewer.py","--request","{request}","--output","{output}"],"timeout_s":900}
```

这是 reviewer 接口，不是已内置的通用视觉模型。命令必须实际看图/视频；不能用模拟成功脚本或文字清单替代。
每次产生 `review_request.json` 和所有帧的 contact sheets；可参考
[Codex 审查技能](../../agentic_skills/ego-third-view-review/SKILL.md)（仓库根目录的 `agentic_skills/ego-third-view-review`）。
审查输出必须包含同一 `binding`、`phase`、`reviewer:{kind,name}`、
`inspected_frames:{"ranges":[[0,最后一帧]]}`、九项 `checks` 和 `decision`。
Accept 必须全帧覆盖、所有项 pass 且无 issue；缺失信息用 unknown/needs_input，不降低门槛。

九项：actor_origin、limb_lengths、torso、contacts、identity、occlusion、camera、timing、background。
返工 issue 格式 `{"code":"dit_timing","evidence":"哪段/哪帧提前或滞后"}`。
仅时序返工还需提供与 SIM **同 ID** 的 `dit_events`（当前被审查 DiT 的时刻、置信度、图像证据）。
先改几何后重生 DiT，不能用局部变速掩盖错物体、缺失动作或相反动作顺序。

## 已纳入的加速与尚待验证项

- 已集成：外部固定版本 Sol-Attn + FirstBlockCache、保持原误差门槛的 consumed-output gate、低临时显存统计、TP 文本编码后 CPU 驻留、同 NUMA UUID 选卡/协作锁、一个 worker 多请求且独立 epoch、冷加载和请求耗时分开。
- worker 在一次 batch 调用内复用模型；调用结束/等待人工审查后释放，不会默认无限占卡。配置实际视觉 reviewer 后可连续批量生成。跨 CLI 调用的长期服务部署尚未启用。
- 原 v7 的 244.65秒加载与803.48秒去噪是旧成功运行证据。新 resident 生命周期 CPU fake 测试证明一次构造/两次调用，**不是新 H3 暖请求测速**。
- 当前接入配置保持4卡 FSDP+Ulysses4和已验证 Triton 路径；不接受未实现的 TP、CuTe、FP8、torch.compile 或 AdaLN 开关，避免“配置写上就算启用”。不同拓扑、完整 CuTe 组合、AdaLN 预计算/融合是下一轮匹配基准候选。
- GPU 空闲不足或没有同 NUMA 组时 `WAITING_GPU`，不会跨 NUMA、降精度、抢占其他进程。协作锁只协调本 pipeline，不能保证其他软件不占卡。
- 分别记录真正的 sparse calls、cache decisions/reuse、每请求各 rank 的 gate。Cache 恰好未复用时不能声称该请求获得缓存收益。

先准备外部固定源码，不下载权重、不修改安装环境：

```bash
python examples/ego_to_third_view/prepare_sol_source.py --output /external/phiagent-sol-pinned
```

固定 `NVlabs/Sana@6fb7eb11c3435555ec6d6adf0d5572d339d2c6eb`，逐文件 Git blob 校验，worker 再验 SHA256。
外部 SGLang H3 模型源码必须匹配该适配器预期哈希；升级后不匹配应停止迁移验证，而不是关掉检查。
保留原生 `quality=lossless` 请求字段只是已验证旧 API 的采样选择，Sol/FBC 本身仍是近似，不是视频无损保证。

## 时序处理

生成前首先保证 EGO、SIM、events/rig 时钟一致并经过全帧审查；H3 参考副本向上补齐 `17*n+5` 帧，
例如288→294，防止原 VAE 把288下裁成277。只补参考尾部、不改 SIM。输出仍取原目标帧数。
这些能防止输入管线时长错误，**不能让软 Ref2VA 条件变成逐帧硬约束**，所以生成后仍要检查事件。

已接受画面只有时间偏移时，用单调 PCHIP 匹配事件，保持首尾和总长；慢处重复、快处跳帧，
不生成新像素或改变手形。拒绝时间反转、缺失事件、低置信度和极端变速，再审查输出。
曲线命中锚点不是独立准确率；应另外测未参与拟合的运动进度/事件。

现有 v8 可单独重放：

```bash
python examples/ego_to_third_view/retime_video.py \
  --dit /data/thirdperson_dit_fixed_torso_v7.mp4 \
  --sim /data/thirdperson_sim_fixed_torso_v35.mp4 \
  --anchors examples/ego_to_third_view/recipes/whiteboard_v35/retime_v8.json \
  --output /data/runs/retime-regression-001
```

## 回归和证据

```bash
python -m unittest discover -s examples/ego_to_third_view/tests -v
```

普通测试不需要 GPU。FFmpeg 可用时执行真实 CPU 编解码、补尾、变速、竖屏审查图和三联测试；
没有 FFmpeg 时明确 skip。batch 状态机测试使用 fake 视频字节，仅验证流程控制，不伪装视频质量。
旧固定场景精确 recipe 用 `pipeline.py render_whiteboard_v35`，仍要求外部原 bundle，参数不能用作新视频默认值。
参见 [有效修正与已知失败](evidence/v35-v8-lessons.md) 和 [验收证据](evidence/accepted-v35-v8.json)。
