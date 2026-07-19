# DepthSync

DepthSync 用拍照帧的高质量深度作为锚点，把短视频的相对 disparity 对齐到同一尺度，降低 Live Photo 在照片和视频之间切换时的虚化跳变。

默认实现面向端侧：拍后阶段只估计每帧两个标量 `scale/offset`，播放阶段只做一次逐像素乘加，不计算稠密光流，也不缓存整段高分辨率深度。

## 轻量算法

1. 将照片 disparity 粗配准并降采样到视频深度分辨率。
2. 在低梯度有效区固定采样，使用分位数初始化、MAD 剔除和一次闭式加权最小二乘拟合锚帧 `scale/offset`。
3. 锚点前后第一帧锁定锚帧参数，保证照片切换邻域不会因自适应更新而退化。
4. 从锚点向前、向后扫描，通过已有编码/ISP block MV 建立少量对应点，逐帧更新 `scale/offset`。
5. 对参数做置信度平滑和单帧变化限幅；MV 缺失或拟合失败时冻结上一帧参数。

测试素材没有 ISP MV，因此验证工具使用 18×32 稀疏 LK 网格模拟输入。该适配器不属于端侧主算法。

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

`offline_prepare` 输出同步深度、逐帧参数、置信度和回退原因。生产环境可以只保存参数表，在播放时调用 `apply_frame`。

## 三视频验证

三个本地视频不会进入 Git。以下命令从每段视频中间截取 3 秒，统一采样为 30 fps / 90 帧，锚帧为索引 45：

```powershell
python -m depthsync.validation testdata\01.mp4 testdata\02.mp4 testdata\03.mp4
```

模型验证使用官方 DepthPro 和 Video Depth Anything Small streaming。模型源码、权重及生成深度均在忽略目录中；精确来源和 SHA-256 见 `config/model-sources.json`。

```powershell
<model-python> tools\run_depth_models.py vda `
  --clip artifacts\clips\01\clip.mp4 --output-dir artifacts\depth\01

<model-python> tools\run_depth_models.py depthpro `
  --anchor artifacts\clips\01\anchor.png --output-dir artifacts\depth\01

python -m depthsync.evaluate
python -m depthsync.depth_visualization
```

评估输出：

- `results/<scene>/comparison.mp4`：固定锚点映射与参数传播的虚化对比；
- `results/<scene>/parameters.csv`：逐帧参数与回退原因；
- `results/<scene>/metrics.json`：尺度、CoC 跳变和耗时；
- `reports/validation.md`：三段视频汇总报告。

每个场景的 `results/<scene>/depth_visualization/` 还包含：

- `depth_comparison.mp4`：RGB、原始 VDA、照片值域下的原始 VDA、同步结果和 DepthPro 锚点五联视频；
- `vda_raw_color.mp4` / `vda_raw_gray.mp4`：原始深度彩色与灰度视频；
- `depthsync_color.mp4` / `depthsync_gray.mp4`：同步后彩色与灰度视频；
- `anchor_comparison.png`：锚帧五联对比图；
- `anchor_*_gray16.png`：使用 `visualization.json` 中值域解释的 16-bit 深度可视化图。

`artifacts/`、`results/`、权重和测试 MP4 均被忽略，只有小型指标报告进入 Git。
