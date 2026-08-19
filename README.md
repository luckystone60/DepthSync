# DepthSync V4.1

DepthSync V4.1 用一张高质量照片深度锚点，把 Video Depth Anything Small 的 90 帧视频深度对齐到同一尺度，同时保持视频原有的时域稳定性。当前仓库只保留 V4.1：不依赖光流模型、人物分割或逐像素局部参数，播放阶段仅执行定长全局单调 LUT。

## 算法概览

1. 读取 3 秒 / 30 fps / 90 帧视频深度，照片锚点固定为第 45 帧。
2. 将 DAv2-Large（默认）或 DepthPro 照片深度缩放到视频深度分辨率。
3. 在锚帧上从 8、16、32、64 个节点中自适应选择全局单调 LUT。
4. 整段固定 LUT 的非线性形状，根据逐帧分布只做受限的全局调整；低相关或大遮挡帧冻结到稳定映射。
5. 序列化逐帧 LUT 参数；播放阶段逐像素查表，不读取照片深度，也不运行额外模型。

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

## V4.1 评估与可视化

```powershell
python -m depthsync.evaluate `
  --scenes 01 02 03 `
  --anchor-model dav2-large `
  --result-root results\v41 `
  --report reports\validation-v41.md

python -m depthsync.depth_visualization `
  --scenes 01 02 03 `
  --anchor-model dav2-large `
  --result-root results\v41
```

20 组验证时把 `--scenes` 改为 `01 02 ... 20`。每个场景输出：

- `v4_depth.npz`：V4.1 浮点深度序列；
- `v4_parameters.npz`：逐帧 LUT 和回退状态；
- `depth_visualization/depth_comparison_dav2_v4.mp4`：VDA 归一化、V4.1、照片锚点三栏对比；
- `depth_visualization/v4_depth_color.mp4` 与 `v4_depth_gray.mp4`：独立深度视频；
- `metrics.json`：锚点误差、时域误差、耗时和参数量。

聚合 20 组结果：

```powershell
python -m depthsync.aggregate_validation `
  --result-root results\v41 `
  --manifest config\pexels-validation-manifest.json `
  --report reports\validation-v41-summary.json
```

## 测试

```powershell
python -m pytest -q
```

模型权重、下载视频、深度缓存和生成视频位于 Git 忽略目录，不提交到远端。
