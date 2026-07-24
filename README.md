# DepthSync

DepthSync 使用 Live Photo 拍照帧的高质量深度作为锚点，把轻量视频深度序列对齐到照片深度的值域与局部结构，同时保持视频模型原有的时域一致性。

当前 V4 面向端侧实现：照片深度只监督一个共享 8 节点单调 LUT 和少量大静态区域的标量 offset，不再传播照片残差图或照片目标层。人物和不可靠区域自动回退到全局 LUT，不需要人物分割 mask。播放阶段不运行深度模型、不计算稠密光流。

端侧处理流程、中间数据规格、参数包、内存/计算量和回退策略见 [`docs/depthsync-v4-design.md`](docs/depthsync-v4-design.md)；可编辑流程图见 [`docs/depthsync-flow.drawio`](docs/depthsync-flow.drawio)。

## V4 算法

1. 把照片 disparity 配准并降采样到视频深度分辨率。
2. 在低梯度有效区域固定采样，人脸区域约占 30%。
3. 使用全局 affine 做鲁棒初始化和回退，并通过分箱中位数及单调回归拟合 8 节点分段线性 LUT。
4. 锚帧确定 LUT 的非线性形状；其他帧只允许对该固定 LUT 做小幅 affine 修正。
5. 结合 block MV、时序深度 MAD 和全片极差，在 128×72 网格估计静态可靠性。
6. 只用锚帧视频深度划分大连通区域；删除面积小、静态支持不足或与可选人脸框相交的区域。
7. 对保留区域只拟合逐帧中位值 offset，使静态区域从第一帧起锁定照片中位尺度。
8. 人物、头发、遮挡边界和不可靠背景全部使用全局 LUT，不需要人物分割 mask。
9. 播放阶段用 block MV 的全局平移传播 U8 区域标签，以 3×3 当前视频深度边缘保护标量修正。
10. 不保存或传播照片残差图、照片目标层；无可靠区域时自动退化为全局 LUT。

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

`offline_prepare` 输出同步深度、共享 LUT、逐帧小 affine、静态区域标签、逐帧区域 offset/shift、置信度和回退原因。生产环境只需保存紧凑参数包，播放时调用 `apply_frame`。

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
- `synced_depth.npz`：V4 LUT + 静态区域标量修正深度；
- `v4_parameters.npz`：LUT、区域标签、逐帧区域 offset/shift、置信度与回退信息；
- `metrics.json`：锚帧误差、切换误差、参数大小和 Python 原型耗时；
- `temporal_errors.npz`：V1/V4 经 block MV 补偿后的逐帧时序差分；
- `depth_visualization/depth_comparison.mp4`：VDA 原始、V1 affine、V4 与 DepthPro 锚点的统一值域四联深度视频；
- `depth_visualization/affine_color.mp4` 和 `depthsync_color.mp4`：V1/V4 独立深度视频；
- `depth_visualization/anchor_comparison.png`：锚帧深度对比图；
- `depth_visualization/worst_temporal_comparison.png`：最大新增跳变帧及其前后帧对比；
- `depth_visualization/anchor_*_gray16.png`：16-bit 灰度深度可视化图。

汇总指标写入 `reports/validation.md`。测试 MP4、模型权重、`artifacts/` 和 `results/` 均被 Git 忽略，不会提交大文件。
