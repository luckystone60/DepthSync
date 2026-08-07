# DAv2-Large 锚点与 20 场景离线验证设计

## 目标

将 V5 离线验证的照片锚点由 DepthPro 替换为官方 Depth Anything V2 Large（DAv2-Large）相对深度，并将验证素材从现有 3 组扩展为 20 组。该工作只评估单帧锚点质量与 V5 对齐收益；DAv2-Large、Pexels 下载和批量离线推理均不属于手机端路径。

## 范围与约束

- 保留现有 `01`–`03`，通过 Pexels API 新增 `04`–`20`，共 20 组。
- 每组统一截取中间 3 秒、30 fps、90 帧，锚帧为第 45 帧。
- 新增源视频优先选择 720p MP4，时长不得少于 3 秒。
- 只下载并缓存 Pexels 许可允许的素材；Git 中只提交来源元数据和哈希，不提交视频、模型、深度、flow 或渲染视频。
- Pexels Key 只从 `PEXELS_API_KEY` 环境变量读取，不写入文件、命令行日志、测试输出或 Git。
- DAv2-Large 权重使用 `depth-anything/Depth-Anything-V2-Large-hf`；其模型卡标注 CC-BY-NC-4.0，因此本次用途严格为离线、非商业仿真验证。将来若进入产品或商业评估，必须重新确认授权并选择可商用替代模型。
- V5 播放仍只使用 V4 参数和局部场；DAv2-Large 不增加端侧模型、内存或拍后 1–2 秒预算。

## 输入、模型与数据流

每组输入为 `testdata/<scene>.mp4`，现有验证工具产出 `artifacts/clips/<scene>/clip.mp4`、RGB 帧与第 45 帧照片模拟图。

DAv2-Large 适配器在隔离模型环境中仅推理锚帧，写入：

```text
artifacts/depth/<scene>/photo_dav2_large_relative.npy
artifacts/depth/<scene>/photo_dav2_large_disparity.npy
artifacts/depth/<scene>/dav2_large.json
```

相对深度通过明确记录的单调方向转换为内部 disparity。若模型输出方向与 VDA 锚帧的可靠样本相关性为负，适配器反转方向；转换前后值域、方向判定、模型仓库 revision、权重 SHA-256、输入尺寸、耗时都写进 JSON。禁止逐帧自动归一化，避免显示或评估掩盖尺度差异。

V5 读取 `photo_dav2_large_disparity.npy` 作为 `P_k`。DepthPro 文件和适配器保留为基线，不参与新的默认 V5 路径。

## 20 组素材集

使用 Pexels 官方视频搜索 API。候选按照五类均衡收集，每类目标 3–4 段：

| 标签 | 目标画面 |
|---|---|
| `walking_turning` | 全身行走、转身，背景有深度层次 |
| `dance_motion` | 舞蹈或大幅肢体运动 |
| `sport_running` | 跑跳、骑行、球类等快速运动 |
| `portrait_close` | 人像近中景、头发和手部细节 |
| `camera_occlusion` | 明显推拉/横移/跟拍，或复杂前景遮挡 |

脚本使用多组英文检索词，并对候选做如下自动筛选：MP4、横向或竖向均接受、最短边至少 720、时长至少 3 秒、Pexels video ID 去重。自动筛选后必须人工快速检查，拒绝无可见人物、过度静态、剪辑跳切主导、主体小于画面 15% 或和既有素材高度相似的条目。

每个入选项写进受版本控制的 `config/pexels-validation-manifest.json`：`scene_id`、`pexels_id`、`page_url`、`photographer`、`query`、`tags`、`selected_file`、`width`、`height`、`duration_seconds`、`sha256`、`license_url`。下载文件位于 Git 忽略的 `testdata/`。

## 标注与评估

每个新增场景必须在 `config/validation-scenes.json` 增加：

- `face_box`：用于兼容既有 V4 评估；
- `subject_box`：人工确认的归一化主体区域；
- `inspection_frames`：至少包括 `0`、`45` 和一个运动/遮挡最显著帧；
- `tail_frames`：最后 15 帧，用于检查结束段时域稳定性。

批处理顺序：准备 clip → VDA Small 流式视频深度 → DAv2-Large 锚点 → SEA-RAFT-S flow → V5（DAv2 锚点）→ 指标 → 可视化。

每个场景导出：

```text
depth_comparison_dav2_v5.mp4   # VDA raw | V4 | V5(DAv2) | DAv2 anchor
dav2_vs_depthpro_anchor.png    # 两种锚点在固定 DAv2 显示值域下的静态对比
field_diagnostics_v5.mp4
frame_XXX_comparison.png
```

20 组总报告同时列出每段值、按标签聚合的中位数/P90、最差三段和异常原因。V5 的原有回退、参数大小和时序门槛保留；新增通过条件是 20 段全部完成可复现推理，且必须显式列出每一段 V5 回退或单段失败，不能用平均值隐藏失败样本。

## 错误处理

- Key 缺失或 Pexels API 限流：停止下载，保留已完成 manifest，不删除已有素材。
- 候选下载或 SHA 校验失败：记录失败项，重新选择候选，不复用不完整文件。
- DAv2 权重缺失、哈希不符、模型输出非有限或方向判定失败：该场景不运行 V5(DAv2)，报告为 `anchor_invalid`；不得悄悄回落到 DepthPro。
- 新场景缺少人工 ROI：可生成深度和视频，但不进入主体指标汇总。
- flow/V5 异常：沿用既有全段 V4 精确回退，并在报告记录。

## 验证

新增单元测试覆盖：Pexels 响应中 720p MP4 选择与去重、API Key 不写入 manifest、DAv2 相对深度方向和无效输出拒绝、V5 默认锚点路径选择、20 场景 manifest 完整性、固定显示值域与聚合报告。

完整验证在模型和素材准备完成后执行：20 组均生成 90 帧输入、DAv2 元数据、V5 参数、指标和比较视频；随后人工播放每段对比视频，优先审查人物边缘、墙面/平面分层、快速运动、遮挡和最差跳变邻帧。
