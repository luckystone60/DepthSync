# DepthSync V4.1

DepthSync V4.1 用一张高质量照片深度锚点，把 Video Depth Anything Small 的 90 帧视频深度对齐到同一尺度，同时保持视频原有的时域稳定性。当前实现以全局单调 LUT 为稳定基线，并在拍摄后的 1–2 秒准备阶段使用小型双向光流拟合 `8×12` 局部 scale/shift 参数场。播放阶段不运行光流，只执行 LUT 和低分辨率参数场采样。

## 算法概览

1. 读取 3 秒 / 30 fps / 90 帧视频深度，照片锚点固定为第 45 帧。
2. 对照片 RGB 与视频锚帧做保守仿射配准；没有明确收益时保持恒等映射。
3. 将 DAv2-Large（默认）或 DepthPro 照片深度映射到视频锚帧坐标。
4. 在锚帧上从 8、16、32、64 个节点中自适应选择全局单调 LUT。
5. SEA-RAFT-S 双向光流只传播可信对应关系；照片深度不直接 warp 到输出帧。
6. 在全局 LUT 结果上拟合 `8×12` 局部 scale/shift，并用遮挡置信度、双向时域正则和逐轨迹参数限幅抑制拖影。
7. 自动选择 `1/0.75/0.5/0.25/0` 局部强度，保证稠密光流补偿后的时域误差不超过预算；可选脸框使用独立保护强度，不需要人物分割 mask。
8. 序列化 LUT 和局部参数；播放阶段不读取照片、不运行光流模型。

详细设计见 [docs/depthsync-v4-design.md](docs/depthsync-v4-design.md)，可编辑流程图见 [docs/depthsync-flow.drawio](docs/depthsync-flow.drawio)。

## 环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy opencv-python pytest
```

单帧/视频深度模型推理需要独立的 PyTorch 环境；模型来源和固定版本记录在 `config/model-sources.json`。DAv2-Large 权重为 CC-BY-NC-4.0，仅用于当前离线仿真验证。

## 运行模型

```powershell
python tools\run_depth_models.py --help
python tools\run_depth_models.py dav2-large --help
python tools\run_depth_models.py video-depth-anything --help
```

批量生成 20 组 DAv2-Large 照片锚点：

```powershell
python tools\run_dav2_validation.py `
  --manifest config\pexels-validation-manifest.json `
  --clip-root artifacts\clips `
  --depth-root artifacts\depth
```

在拍摄后准备阶段生成 SEA-RAFT-S 双向光流缓存（评估前执行一次）：

```powershell
python tools\run_flow_model.py `
  --scenes 01 02 03 `
  --repository third_party\SEA-RAFT `
  --checkpoint models\sea_raft_s.safetensors `
  --device cuda
```

仓库、配置和权重的固定 revision/SHA256 见 `config/model-sources.json`。该入口会输出相邻帧双向 flow、置信度、切镜标记与耗时；播放阶段不加载这些缓存，也不依赖 PyTorch。

## V4.1 评估与可视化

```powershell
python -m depthsync.evaluate `
  --scenes 01 02 03 `
  --anchor-model dav2-large `
  --flow-root artifacts\flow `
  --result-root results\v41-local `
  --report reports\validation-v41-local.md

python -m depthsync.depth_visualization `
  --scenes 01 02 03 `
  --anchor-model dav2-large `
  --result-root results\v41-local
```

20 组验证时把 `--scenes` 改为 `01 02 ... 20`。每个场景输出：

- `v4_depth.npz`：V4.1 浮点深度序列；
- `v4_parameters.npz`：逐帧 LUT 和回退状态；
- `v41_local_depth.npz`：通过时域安全门后的局部增强深度；
- `v41_local_parameters.npz`：FP16 局部参数、置信度与深度引导；
- `depth_visualization/depth_comparison_dav2_v41_local.mp4`：VDA、全局 LUT、局部增强、照片锚点四栏对比；
- `depth_visualization/v4_depth_color.mp4` 与 `v4_depth_gray.mp4`：独立深度视频；
- `depth_visualization/v41_local_depth_color.mp4` 与 `v41_local_depth_gray.mp4`：局部增强深度视频；
- `compare_videos/`：所有场景四联对比视频的集中目录；
- `metrics.json`：锚点误差、时域误差、耗时和参数量。

聚合 20 组结果：

```powershell
python -m depthsync.aggregate_validation `
  --result-root results\v41-local `
  --manifest config\pexels-validation-manifest.json `
  --report reports\validation-v41-summary.json
```

## 测试

```powershell
python -m pytest -q
```

模型权重、下载视频、深度缓存和生成视频位于 Git 忽略目录，不提交到远端。
