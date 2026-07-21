# DepthSync

DepthSync 使用 Live Photo 拍照帧的高质量深度作为锚点，把轻量视频深度序列对齐到照片深度的值域与局部结构，同时保持视频模型原有的时域一致性。

当前 V3 面向端侧实现：每帧使用一个 8 节点单调 LUT，并在照片锚点附近叠加一个 16×9 的低分辨率残差网格。播放阶段不运行深度模型、不计算稠密光流。

## V3 算法

1. 把照片 disparity 配准并降采样到视频深度分辨率。
2. 在低梯度有效区域固定采样，人脸区域约占 30%。
3. 使用全局 affine 做鲁棒初始化和回退，并通过分箱中位数及单调回归拟合 8 节点分段线性 LUT。
4. 在锚帧计算 `照片深度 - LUT(视频深度)`，裁剪异常值后压缩为 16×9 残差网格。
5. 从锚点向前、向后扫描，使用已有编码器或 ISP block MV 传播 LUT 与残差网格；LUT 做置信度平滑和单帧限幅。
6. 残差只在锚点前后 15 帧内启用，并按余弦权重衰减，避免把静态照片结构长期固化到视频中。
7. 播放阶段计算 `D' = LUT(D) + upsample(residual_grid)`。

验证素材没有端侧 ISP MV，因此工具使用 18×32 稀疏 LK 网格模拟输入。该运动估计适配器不属于端侧主算法。

## 安装和测试

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

核心接口：

```python
sync = DepthSync(DepthSyncConfig(depth_mode="disparity"))
result = sync.offline_prepare(video_depths, photo_depth, anchor_index, motion, face_box)
display_depth = sync.apply_frame(video_depth, result.parameters[frame_index])
```

`offline_prepare` 输出同步深度、逐帧 LUT、残差网格、兼容的 affine 参数、置信度和回退原因。生产环境只需保存参数表，播放时调用 `apply_frame`。

## 三视频验证

以下命令从项目中的三个视频各截取中间 3 秒，统一为 30 fps / 90 帧，锚帧索引为 45：

```powershell
python -m depthsync.validation testdata\01.mp4 testdata\02.mp4 testdata\03.mp4
python -m depthsync.evaluate
python -m depthsync.depth_visualization
```

模型深度使用官方 DepthPro 和 Video Depth Anything Small streaming 模式生成，模型来源及校验信息见 `config/model-sources.json`。

每个场景的 `results/<scene>/` 包含：

- `affine_depth.npz`：V1 全局 affine 基线深度；
- `synced_depth.npz`：V3 LUT + 残差网格深度；
- `v3_parameters.npz`：逐帧 LUT、残差网格、置信度与回退信息；
- `metrics.json`：锚帧误差、切换误差、参数大小和 Python 原型耗时；
- `depth_visualization/depth_comparison.mp4`：VDA 原始、V1 affine、V3 与 DepthPro 锚点的统一值域四联深度视频；
- `depth_visualization/affine_color.mp4` 和 `depthsync_color.mp4`：V1/V3 独立深度视频；
- `depth_visualization/anchor_comparison.png`：锚帧深度对比图；
- `depth_visualization/anchor_*_gray16.png`：16-bit 灰度深度可视化图。

汇总指标写入 `reports/validation.md`。测试 MP4、模型权重、`artifacts/` 和 `results/` 均被 Git 忽略，不会提交大文件。
