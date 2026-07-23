# DepthSync

DepthSync 使用 Live Photo 拍照帧的高质量深度作为锚点，把轻量视频深度序列对齐到照片深度的值域与局部结构，同时保持视频模型原有的时域一致性。

当前 V3.7 面向端侧实现：使用锚帧确定的 8 节点单调 LUT，每帧只更新两个受限修正参数；人物运动走廊只使用固定形状 LUT 和全片恒定 offset，非人物动态区域可叠加 16×9 低分辨率残差，可靠静态深度平面使用 128×72 照片目标层。播放阶段不运行深度模型、不计算稠密光流。

端侧处理流程、中间数据规格、参数包、内存/计算量和回退策略见 [`docs/depthsync-v37-design.md`](docs/depthsync-v37-design.md)；可编辑流程图见 [`docs/depthsync-flow.drawio`](docs/depthsync-flow.drawio)。

## V3.7 算法

1. 把照片 disparity 配准并降采样到视频深度分辨率。
2. 在低梯度有效区域固定采样，人脸区域约占 30%。
3. 使用全局 affine 做鲁棒初始化和回退，并通过分箱中位数及单调回归拟合 8 节点分段线性 LUT。
4. 在锚帧计算 `照片深度 - LUT(视频深度)`，裁剪异常值后压缩为 16×9 残差网格。
5. 锚帧确定 LUT 的非线性形状；其他帧只允许对该固定 LUT 做小幅 affine 修正，避免逐帧重拟合 LUT 节点造成闪动。
6. 从锚点向前、向后扫描，使用已有编码器或 ISP block MV 建立对应；LUT 修正在锚点附近渐进解锁。
7. 16×9 残差网格按完整 block MV 位移传播，并在锚点前后 30 帧内缓慢衰减。
8. 结合 block MV、时序深度 MAD，以及“3 帧时域中值 + 全片极差”，在 128×72 网格捕获仅在片头或片尾短暂出现的人物占用。
9. 将与人物运动走廊相交的照片深度连通平面整体排除，避免轮廓形孔洞或矩形保护区在背景留下拖影、竖线和网格。
10. 人物走廊只使用固定形状 LUT，并把锚帧主体的中位尺度偏差折算成全片恒定的小 offset；16×9 空间残差不进入人物走廊。
11. 可靠静态平面仍完成到真实深度边界，01 左墙保持照片尺度且不重新产生宽回退带。
12. 静态照片目标层使用 128×72；8-bit 对比视频使用固定亚 LSB 抖动抑制可视化伪轮廓，浮点同步深度不添加噪声。
13. 播放阶段用一次 7×7 深度形态学梯度保护动态残差；静态照片层仅作用于离线确认安全的完整深度平面。

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

`offline_prepare` 输出同步深度、逐帧 LUT、LUT 的小幅 affine 修正、残差网格、静态掩码/照片目标层、置信度和回退原因。生产环境只需保存参数表，播放时调用 `apply_frame`。

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
- `temporal_errors.npz`：V1/V3 经 block MV 补偿后的逐帧时序差分；
- `depth_visualization/depth_comparison.mp4`：VDA 原始、V1 affine、V3 与 DepthPro 锚点的统一值域四联深度视频；
- `depth_visualization/affine_color.mp4` 和 `depthsync_color.mp4`：V1/V3 独立深度视频；
- `depth_visualization/anchor_comparison.png`：锚帧深度对比图；
- `depth_visualization/worst_temporal_comparison.png`：最大新增跳变帧及其前后帧对比；
- `depth_visualization/anchor_*_gray16.png`：16-bit 灰度深度可视化图。

汇总指标写入 `reports/validation.md`。测试 MP4、模型权重、`artifacts/` 和 `results/` 均被 Git 忽略，不会提交大文件。
