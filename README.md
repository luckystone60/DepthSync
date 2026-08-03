# DepthSync

## DAv2-Large 离线锚点验证

V5 的 20 场景扩展验证默认使用 `Depth-Anything-V2-Large` 生成照片锚点；该模型只用于离线仿真，不会进入端侧、播放或拍后准备路径。模型卡使用 `CC-BY-NC-4.0`，因此当前用途限于非商业验证。DepthPro 仍保留为基线。

```powershell
C:\Users\jiao\Documents\DepthSync\.venv-models\Scripts\python.exe tools\run_depth_models.py dav2-large `
  --anchor artifacts\clips\01\anchor.png `
  --reference-vda artifacts\depth\01\video_disparity.npz `
  --output-dir artifacts\depth\01 `
  --revision 7581137eff8d4e94f6e796d3baea0e9fa79b22d2
```

DAv2 输出保存在 Git 忽略的 `artifacts/depth/<scene>/`，包含原始相对深度、方向统一后的 disparity 和完整推理元数据。

当前效果优先版本为 V5：以 V4 全局单调 LUT 作为稳定基线，在照片锚帧拟合 `36×64` 局部 affine 场，并通过 SEA-RAFT-S 双向光流在拍后准备阶段向整段视频传播。播放阶段不运行模型，低置信度区域严格回退 V4。

- 端侧算法、数据规格、内存/算力和回退策略：[`docs/depthsync-v5-algorithm.md`](docs/depthsync-v5-algorithm.md)
- 可编辑处理流程图：[`docs/depthsync-flow.drawio`](docs/depthsync-flow.drawio)
- 流程图预览：[`docs/depthsync-flow.png`](docs/depthsync-flow.png)
- 三段 90 帧验证报告：[`reports/validation-v5.md`](reports/validation-v5.md)

V5 验证与可视化：

```powershell
python tools\run_flow_model.py --scenes 01 02 03 --device cuda
python -m depthsync.evaluate --algorithm-version v5 --flow-root artifacts\flow --result-root results\v5 --report reports\validation-v5.md
python -m depthsync.depth_visualization --flow-root artifacts\flow --result-root results\v5
```

每个场景会生成 `depth_comparison_v5.mp4`、`field_diagnostics_v5.mp4`、V5 彩色/灰度深度视频和指定关键帧 PNG。模型、光流、深度缓存及视频均位于 Git 忽略目录，不提交到远端。

DepthSync 使用 Live Photo 拍照帧的高质量深度作为锚点，把轻量视频深度序列对齐到照片深度的值域与局部结构，同时保持视频模型原有的时域一致性。

当前 V4 面向端侧实现：照片深度只监督一个全局单调 LUT，不再使用运行时空间掩码、区域标签、照片残差图或照片目标层。离线在 8/12/16 个有效节点中自动选择误差最小的 LUT 形状，并统一序列化为 16 节点。播放阶段不运行深度模型、不需要人物分割 mask，也不计算光流。

端侧处理流程、中间数据规格、参数包、内存/计算量和回退策略见 [`docs/depthsync-v4-design.md`](docs/depthsync-v4-design.md)；可编辑流程图见 [`docs/depthsync-flow.drawio`](docs/depthsync-flow.drawio)。

## V4 算法

1. 把照片 disparity 配准并降采样到视频深度分辨率。
2. 在低梯度有效区域固定采样，人脸区域约占 30%。
3. 使用全局 affine 做鲁棒初始化和回退。
4. 分别拟合 8、12、16 个有效节点的单调 LUT；以锚帧全图误差和人脸 ROI 误差之和选择最优形状。
5. 将选中的 LUT 重采样成固定 16 节点，方便端侧使用定长结构。
6. 锚帧确定 LUT 的非线性形状；其他帧只允许对该固定形状做小幅 affine 修正。
7. 从锚点向前、向后扫描，使用 block MV 稀疏对应约束逐帧参数，锚点附近渐进解锁。
8. 所有像素使用同一条单调映射；不传播任何空间参数，因此不会产生区域边缘缝隙、轮廓或网格。

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

`offline_prepare` 输出同步深度、共享 16 节点 LUT、逐帧小 affine、置信度和回退原因。生产环境只需保存定长参数表，播放时调用 `apply_frame`。

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
- `synced_depth.npz`：V4 自适应全局单调 LUT 深度；
- `v4_parameters.npz`：LUT、逐帧小 affine、置信度与回退信息；
- `metrics.json`：锚帧误差、切换误差、参数大小和 Python 原型耗时；
- `temporal_errors.npz`：V1/V4 经 block MV 补偿后的逐帧时序差分；
- `depth_visualization/depth_comparison.mp4`：VDA 原始、V1 affine、V4 与 DepthPro 锚点的统一值域四联深度视频；
- `depth_visualization/affine_color.mp4` 和 `depthsync_color.mp4`：V1/V4 独立深度视频；
- `depth_visualization/anchor_comparison.png`：锚帧深度对比图；
- `depth_visualization/worst_temporal_comparison.png`：最大新增跳变帧及其前后帧对比；
- `depth_visualization/anchor_*_gray16.png`：16-bit 灰度深度可视化图。

汇总指标写入 `reports/validation.md`。测试 MP4、模型权重、`artifacts/` 和 `results/` 均被 Git 忽略，不会提交大文件。

## V5 光流准备阶段

V5 保留 V4 全局 LUT 作为基础深度，只在拍摄完成后的准备阶段运行
SEA-RAFT-S。模型计算相邻帧双向光流和不确定度，DepthSync 再通过前后向
一致性过滤遮挡，把锚帧的 `64×36` 局部 scale/offset/confidence 参数场沿
可信轨迹传播。播放阶段不运行模型；置信度为零时逐像素回退 V4。

官方源码、权重版本和 SHA-256 固定在 `config/model-sources.json`。准备环境：

```powershell
git clone https://github.com/princeton-vl/SEA-RAFT.git third_party\SEA-RAFT
git -C third_party\SEA-RAFT checkout 9137517ba24e628442aec097d3afe71d03503b75
python -c "from pathlib import Path; from shutil import copy2; from huggingface_hub import hf_hub_download; Path('models').mkdir(exist_ok=True); copy2(hf_hub_download(repo_id='MemorySlices/Tartan-C-T-TSKH-spring540x960-S', filename='model.safetensors', revision='31b9b4b711bd2d0d38cb99d93ba144beb2dc92'), 'models/sea_raft_s.safetensors')"
Get-FileHash -Algorithm SHA256 models\sea_raft_s.safetensors
```

期望权重哈希为
`D6A75E47F2630BA6C354CE84A322E24C3D9DEF668A6956F340A054B9A3211908`。
生成三段 90 帧光流缓存：

```powershell
python tools\run_flow_model.py --scenes 01 02 03 --device cuda
```

模型权重、第三方源码和生成的光流仍位于 Git 忽略目录，不进入普通提交。
