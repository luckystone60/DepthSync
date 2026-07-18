# DepthSync

DepthSync 用一张高质量拍照深度图作为时间锚点，把低分辨率、尺度漂移的视频深度序列统一到拍照深度的尺度，并降低 Live Photo 人像虚化在照片/视频切换时的跳变。实现是无需训练的轻量方案。

## 算法

1. 默认将深度转成逆深度（disparity）工作，以便用仿射模型表达常见单目深度的 scale/shift 歧义。
2. 在照片对应的视频帧上，用分位数初始化 + Huber IRLS 拟合 `D_photo ≈ a D_video + b`。
3. 以照片帧为中心向前、向后传播。若提供 RGB，使用稠密光流把上一张同步深度 warp 到当前帧，并通过前后向一致性与亮度残差估计置信度；无 RGB 时使用保守的同位置传播。
4. 每帧先对传播结果重新做鲁棒 scale/shift 标定，再按光流置信度与深度分歧自适应时域融合。真实运动/遮挡区域更多相信当前视频深度，稳定区域更多继承锚点尺度。
5. 照片锚帧注入受控的高频残差，提升边缘细节。非锚帧的细节随光流逐帧传播，遮挡处会自动衰减。

这套方法适合短时、运动有限的 Live Photo。它不会凭空恢复视频中照片不可见区域的细节；若需要强泛化的细节生成，可在本方案输出后增加一个小型 refinement 网络。

## 使用

```powershell
python -m pip install -e .
depthsync --video-depth video_depth.npy --photo-depth photo_depth.npy --anchor 15 --rgb-dir frames --output synced.npz
```

- `video_depth.npy`: `[T,H,W]` 浮点数组。
- `photo_depth.npy`: `[H2,W2]` 浮点数组；输出采用此分辨率。
- `frames/`: 可选，按文件名排序的 RGB 帧，数量必须等于 T。
- 输出包含 `depths/scales/offsets/confidences`。
- 若输入本身是 inverse depth/disparity，添加 `--mode disparity`。

生产接入时应保持深度单位一致、明确无效值（本实现把非有限值及 `<=0` 视为无效），并建议用人像 mask 限制拟合区域，避免大面积动态背景主导标定。

## 测试

```powershell
python -m unittest discover -s tests -v
```
