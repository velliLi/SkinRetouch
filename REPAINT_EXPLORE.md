# 局部重绘美肤：离线可行性实测

目标：在这台机器**完全离线**的条件下，验证"用扩散模型对瑕疵区做局部重绘（inpaint）"
能否作为 SkinRetouch 的一条美肤途径，并与现有 CPU 保边滤波管线做定量对比。

结论按证据分级，全部数字来自本仓库 `tests/probe_repaint_*.py` 的可复现实验。

---

## 0. 结论先行

1. **可行，但只能作为"定点瑕疵清除"，替代不了 CPU 管线。**
   在注入式痘点/红斑基准上，瑕疵区残差 `resid` 从坏图 `7.95` 降到 `4.48`（纯重绘）
   或 `5.00`（CPU 磨皮后重绘），而现有管线是 `6.71`；同时纯重绘不会降低整片皮肤的
   高频噪声（`hf` 9.56 → 9.27，几乎不动），因为它只改掩膜内像素。
   两者是**互补关系**：CPU 管线管全皮肤去噪，重绘管定点清除残留。

2. **最优组合是"先 CPU 磨皮，再对残留瑕疵重绘"**：
   `hf 7.36`（最接近干净基准 5.93）、`resid 5.00`、`SSIM 0.9563`，
   代价 `0.8s + 9.9s`。

3. **本机可用的重绘引擎只有一个**：SDXL 9 通道 inpaint checkpoint，
   经 ComfyUI 核心节点驱动。真正的"小模型"修复网（LaMa / MI-GAN / AOT-GAN / SD1.5-inpaint）
   在本机不存在，且机器无外网，装不上。

4. **最大工程坑不是模型，是掩膜分辨率**：掩膜会被线性插值到 latent 尺度（1/8），
   半径 3–8 px 的痘点在 latent 上只剩亚像素权重，`KSamplerX0Inpaint` 于是把原 latent
   大部分混回输出，痘点原封不动。掩膜必须先膨胀（实测 `grow=8 px`）再重绘。

5. **对真实照片的弱瑕疵，模型保守到几乎不动**：两张 20 MP 真实人像上，
   掩膜内平均改动只有 `3.3 / 3.6` 灰度级（与 VAE 往返误差 `3.2` 同量级）。
   ⇒ 安全性没问题（不会毁图），但对轻微真实瑕疵的收益接近于零。

6. **大掩膜重绘会坏画面，小掩膜不会**。把掩膜换成整张皮肤（占比 40%）后，
   `SSIM` 从 0.954 掉到 **0.899**、瑕疵区残差反而从 7.95 恶化到 **8.39**，
   多块拼接还在接缝行留下 **4.66** 的梯度跳变（相邻行 2.2–3.0）。
   而稀疏痘点掩膜（占比 4.8%）分块重绘后接缝梯度与坏图**逐位相同**（3.6150 vs 3.6172）。
   ⇒ `HANDOFF.md` 的"重绘会漂"担忧，在大掩膜下成立、在痘点掩膜下不成立。

7. **已落进网页端，默认关闭**（`repaint` 开关，见第 11 节）。端到端实测：
   瑕疵区残差 `4.80 → 4.10`、高频 `6.46 → 5.46`，掩膜外像素逐字节不变，
   一张 896×1152 竖图约 20 秒（命中 2 块）。
   **短板已定位在自动检测而非引擎**：自动掩膜只覆盖注入痘点的 45%；
   换真值掩膜，同一个引擎能把残差从磨皮后的 `6.20` 压到 `4.53`。

---

## 1. 离线引擎盘点（为什么只剩 SDXL inpaint）

| 候选 | 状态 |
| --- | --- |
| LaMa / MI-GAN / AOT-GAN | 本机不存在，`ComfyUI/models/onnx` 与 `models/diffusers` 均为空 |
| SD1.5 inpaint | 本机不存在 |
| SDXL inpaint（9ch） | **存在 3 个**：`utopianPony_v1Inpainting.safetensors`(6.94 GB)、`pornmaster_proSDXLV7-inpainting.safetensors`(7.11 GB)、`Pony Realism 🔮_v2.1 VAE.safetensors`(7.11 GB) |
| LCM / Turbo 加速 LoRA | `models/loras` 内无，无法把 20 步压到 4 步 |

网络实测：`https://huggingface.co/`、`https://github.com/` 均超时（curl 返回 000），
`~/.cache/huggingface/hub` 为空。⇒ 任何需要下载权重的路线在本机不成立。

`E:\comfyui\ComfyUI\models` 全目录盘点（`find` + 文件数）确认：`onnx/`、`diffusers/`、
`unet/`、`latent_upscale_models/`、`detection/` 全为 0 文件；`loras/` 15 个文件共 2.6 GB
**全是内容类微调，没有 LCM/Turbo**；`model_patches/` 只有 Z-Image 控制网（与 SDXL 不通用）。

`diffusers` 直连路线也已排除：`StableDiffusionXLInpaintPipeline.from_single_file()`
在 `config=None` 时会调 `fetch_diffusers_config()` 去 HF 取 pipeline 配置，
离线必炸（`huggingface_hub.errors.LocalEntryNotFoundError`，
栈落在 `diffusers/loaders/single_file.py:439 _download_diffusers_model_config_from_hub`）。

因此推理引擎用 **ComfyUI 0.29.0 自身核心节点**（这台机器本来就有的本地引擎，
零配置猜测，将来要落进节点也是同一套节点语义）。

---

## 2. 引擎搭建（可复现）

```bash
E:\comfyui\python_embeded\python.exe main.py --port 8189 --listen 127.0.0.1 \
  --disable-auto-launch --disable-all-custom-nodes \
  --output-directory E:\SkinRetouch\tests\output\comfy_tmp
```

`--disable-all-custom-nodes` 是必需的：不禁用时 ComfyUI-Manager 会在加载阶段尝试联网，
启动卡死。

节点图（`tests/probe_repaint_local.py:run_repaint` 组装，走 HTTP `/prompt`）：

```
CheckpointLoaderSimple ─┬─> CLIPTextEncode(pos) ─┐
                        ├─> CLIPTextEncode(neg) ─┤
                        ├─> (vae) ───────────────┤
LoadImage(tile) ────────┼────────────────────────┤
LoadImageMask(mask) ────┴─> InpaintModelConditioning ─> KSampler(euler/normal, cfg 5.0)
                                                          └─> VAEDecode ─> SaveImage
```

关键语义核对（避免掩膜反向这类静默错误）：

- `nodes.py:1818 LoadImageMask.load_image_mask` 取通道值后**直接返回，不做反相**。
- `comfy/samplers.py:633-641 KSamplerX0Inpaint.__call__`：
  `latent_mask = 1. - denoise_mask`，末尾 `out = out*denoise_mask + latent_image*latent_mask`
  ⇒ **掩膜 1 = 要重绘**，0 = 保持原样。
- `nodes.py:470-500 InpaintModelConditioning` 返回的 `samples` 是**原图** VAE latent，
  `concat_latent_image` 是"掩膜区涂 0.5 灰"的 latent（9 通道 UNet 的额外输入）。

---

## 3. 致命细节：掩膜在 latent 尺度上会衰减

掩膜从像素尺寸被 `comfy/utils.py:1312 reshape_mask` 用 **bilinear** 插值缩到 latent
尺寸（H/8）。实测（本地 torch 复算）：

| 掩膜 | latent 上 `>0.9` 的占比 |
| --- | --- |
| 圆盘 r=100 px @896² | 3.9% |
| 痘点 r=6 px（原样） | 0.0% |
| 痘点 r=6 px，膨胀 8 px | 0.2% |

痘点中心在 latent 上只剩 1 个亚像素，`denoise_mask` 权重不足 ⇒ 采样器把原 latent 混回来
⇒ 痘点在输出里**原封不动**。这就是"改了 denoise 却看不到任何变化"的真实原因
（早期实验中 `denoise 0.30/0.45/0.60` 三档输出互差仅 0.25 灰度级）。

处理：`probe_repaint_local.prep_mask()` 先椭圆核膨胀 `grow` 像素，再高斯模糊，
最后按峰值归一化到 255，保证 latent 尺度上有完整权重。

**膨胀半径必须扫**，不是越大越好：

| grow | 掩膜占比 | resid（越低越好） |
| --- | --- | --- |
| 0 | 0.38% | 6.49 |
| 4 | 1.09% | 4.99 |
| **8** | **2.13%** | **4.65** |
| 16 | 5.09% | 5.28 |

grow=16 反而变差：重绘面积过大，模型自由发挥开始偏离原皮肤。

---

## 4. 实测数据

基准：`tests/probe_repaint_local.py` 复用 `tests/blemish_test.py` 的注入协议
（26 个痘点 r∈[3,8] + 10 片低频红斑 + σ=3 颗粒），注入自检
`inject_matches_blemish_test: True`（`clean hf = 5.93`，与合入基线 5.9 一致）。
资产 `tests/assets/t01_headshoulders_135_00001_.png`，tile 896×896，瑕疵掩膜 4094 px。

| 方案 | resid↓ | hf（全皮肤） | SSIM | 耗时 |
| --- | --- | --- | --- | --- |
| 干净基准 | — | **5.93** | 1.0000 | — |
| 坏图（注入后） | 7.95 | 9.56 | 0.9539 | — |
| CPU 管线 `skin_retouch(0.75/0.35/0.7/0.5/0.5)` | 6.71 | 7.51 | **0.9569** | 0.8 s |
| 纯重绘 grow8 d0.60 s20 | **4.48** | 9.27 | 0.9536 | 18.0 s |
| 纯重绘 grow8 d0.60 s12 | 4.52 | 9.27 | 0.9536 | 11.9 s |
| 纯重绘 grow8 d0.60 s8 | 4.64 | 9.27 | 0.9536 | 9.9 s |
| **CPU 磨皮 → 重绘** grow8 d0.60 s8 | 5.00 | **7.36** | 0.9563 | 10.7 s |

读法：
- `resid` = 瑕疵区与干净图的平均绝对差 ⇒ 重绘把痘点真正修掉了，CPU 管线只能靠模糊压低。
- `hf` = 全皮肤 Laplacian 能量 ⇒ 重绘**几乎不动**它（9.56→9.27），因为掩膜只覆盖 2.13%，
  其余 97.87% 的颗粒噪声一个像素都没碰。这是局部重绘的物理上限，不是参数没调好。
- 组合方案是唯一同时把两个指标都拉向干净基准的：`hf 7.36` + `resid 5.00`。

**denoise 几乎不影响结果**（grow=8, steps=20）：

| denoise | 0.35 | 0.50 | 0.60 | 0.65 | 0.80 | 1.00 |
| --- | --- | --- | --- | --- | --- | --- |
| resid | 4.46 | 4.55 | 4.48 | 4.52 | 4.65 | 4.73 |

原因：掩膜极小且任务是"补成周围皮肤"，模型在任何强度下都收敛到同一个解；
`KSamplerX0Inpaint` 又把掩膜外锁死，所以 denoise 只影响掩膜内的重绘火候。
⇒ **真正要调的是掩膜，不是 denoise**。取 `0.60` 即可。

**步数**：8 步与 20 步的 `resid` 差 0.16，耗时差 8 s ⇒ 生产取 8 步。

---

## 5. 视觉证据

痘点放大（5×，6 个最显著的注入瑕疵）：

![痘点放大对照](tests/output/repaint_zoom.png)

- 第 1 行发际线红斑：`repaint` 完全清除，**发丝一根没断**；`skin_retouch` 只压暗、留痕。
- 第 2/3/4/5 行：`repaint` 都把红斑清干净且没有塑料感；`retouch+repaint` 肤色最贴近 reference。
- 第 6 行发丝交界：三列都保住发丝，`repaint` 有轻微色调偏移（VAE 往返所致）。

宽视（400²）：

![宽视对照](tests/output/repaint_compare.png)

真实照片（无注入，掩膜由 CPU 管线的红色 midband 响应自动生成）：

![真实照片对照](tests/output/real_DSC_4081_zoom.png)

- `DSC_4081.JPG`：掩膜 1.15%，掩膜内平均改动 `3.32`、p99 `11`，掩膜外 VAE 漂移 `3.99`。
- `DSC_7300.JPG`：掩膜 0.42%，掩膜内平均改动 `3.62`、p99 `13`，掩膜外漂移 `4.46`。
- 视觉上原图与重绘几乎一致，暗斑仍在 ⇒ **真实弱瑕疵场景下这套模型选择"不动"**。

---

## 6. 与现有 CPU 管线的定位

`HANDOFF.md` 里"不用扩散模型重绘"的担忧是**人脸身份漂移**。本轮实测把这条约束切成了两半：

- **该担忧成立的范围**：大掩膜（覆盖五官/轮廓）重绘会重画结构，确实会漂。
- **本轮用途不落在该范围**：痘点掩膜只覆盖 2.13% 的皮肤像素，且 `KSamplerX0Inpaint`
  把掩膜外锁死为原 latent（实测掩膜外漂移 `4.06` 就是纯 VAE 往返量级，无结构改动）。
  发丝、下颌边缘在放大图里保持完整。

所以两者不是替代关系：

| | CPU 管线 | 局部重绘 |
| --- | --- | --- |
| 作用范围 | 全皮肤 | 仅掩膜内（≈2%） |
| 强项 | 全片去噪、纹理可控、0.8 s | 定点清除，接近真值（resid 减半） |
| 弱项 | 靠模糊，大瑕疵只能压暗 | 不动掩膜外；单 tile 10 s；显存 2 GiB 常驻 |
| 弱瑕疵 | 有效（平滑） | 几乎无动作 |

---

## 7. 落地形态与成本

节点内集成的可行形态：

1. SkinRetouch 先用现有 CPU 检测得到瑕疵掩膜（`retouch_channel` 里的 `midband` 就是现成的
   瑕疵频带响应，`probe_repaint_real.py:redness_midband()` 已复用）。
2. 掩膜膨胀 8 px → 只对含瑕疵的 tile 调本地 ComfyUI HTTP（`/prompt` → `/history` → `/view`）。
3. 结果按软 alpha 融回原图，掩膜外像素原样返回。

成本：896² tile / 8 步 ≈ **10 s**（RTX 3060 Laptop 6 GB），常驻显存约 2 GiB
（加载后 `vram_free` 从 6.0 降到 4.04 GiB，`torch_vram_free` 仅 0.03 GiB ⇒ 碎片紧张）。
全图 896×1152 需 2 tile ≈ 20 s。

---

## 8. 多块拼接：小掩膜安全，大掩膜会裂

20 MP 的照片必须切成多个 896 px tile，每个 tile 各付一次 VAE 往返，色调就会漂。
`tests/probe_repaint_tiles.py` 在同一帧上切 640 px tile（重叠、权重归一化融合）实测：

| 掩膜 | 覆盖像素 | 瑕疵区残差 | SSIM | 接缝行梯度（上/缝/下） | 总耗时 |
| --- | --- | --- | --- | --- | --- |
| 稀疏痘点 | 4.8% | **4.62** | **0.9532** | 3.62 / 2.98 / 3.20（= 坏图，无改动） | 24.7 s / 4 tile |
| 整张皮肤 | 39.5% | 8.39 | 0.899 | 2.98 / **4.66** / 2.18 | 24.8 s / 4 tile |
| 干净参考 | — | — | 1.000 | 2.80 / 2.25 / 2.30 | — |

- 稀疏掩膜：接缝行的梯度与坏图**逐位相同**（3.6150 vs 3.6172），说明接缝处一个像素都没重绘
  ⇒ 分块重绘对"定点瑕疵清除"是安全的，可以放心切 tile。
- 大掩膜：接缝行梯度 4.66，比相邻行高一倍，`SSIM` 掉 0.055，
  视觉上皮肤整体变亮、横向分层可见（![接缝](tests/output/tiles_seam.png)）
  ⇒ 这条路只能走小掩膜。


## 9. 未验证项与风险

- **未测人脸身份度量**：本机无 `insightface` 之外的 face-embedding 比对脚本
  （`~/.insightface/models` 为空，无 ArcFace/buffalo 权重），也没有 `lpips`/`pyiqa`。
  身份保真只有视觉判断 + 掩膜外漂移（`4.06`，等于纯 VAE 往返）作代理证据。
- **未测真实强瑕疵**：手上真实照片都是干净皮肤，给不出"真实痘/斑可修"的定量结论。
- **显存竞争**：ComfyUI 常驻约 2 GiB 会挤压同机的其它模型；6 GB 卡上并行跑别的大模型会 OOM。
- **无加速方案**：本机没有 LCM/Turbo LoRA，8 步 / 10 s per tile 就是下限。
- **验证集单一**：注入基准只跑了 `t01`，`t07`/`t05` 未跑重绘（CPU 基线的三行数字已复跑通过）。

---

## 10. 复现

```bash
# 1) 起 ComfyUI（后台）
E:\comfyui\python_embeded\python.exe E:\comfyui\ComfyUI\main.py --port 8189 \
  --listen 127.0.0.1 --disable-auto-launch --disable-all-custom-nodes \
  --output-directory E:\SkinRetouch\tests\output\comfy_tmp

# 2) 注入基准 + 重绘（参数扫描）
E:\comfyui\python_embeded\python.exe tests/probe_repaint_local.py \
  --tile 896 --mask-grow 8 --mask-blur 4 --denoise 0.60 --steps 8 [--pre-pipeline]

# 3) 真实照片（掩膜由 CPU 瑕疵响应生成）
E:\comfyui\python_embeded\python.exe tests/probe_repaint_real.py \
  --asset DSC_4081.JPG --denoise 0.60 --pct 97

# 4) 出对照图
E:\comfyui\python_embeded\python.exe tests/probe_repaint_zoom.py
```

产物：`tests/output/repaint_report.json`、`real_DSC_*_report.json`、
`repaint_zoom.png`、`repaint_compare.png`、`real_DSC_*_zoom.png`。

`tests/blemish_test.py` 的三行 hf-energy 基线未被改动（`inject_matches_blemish_test: True`）。

---

## 11. 网页端落地与自动检测的实测

### 11.1 形态

`server/repaint.py` 是纯标准库客户端，把 ComfyUI 当 IPC 用；`server/jobs.py` 在 CPU 磨皮之后
调它一过。三个参数进 `PARAM_SPECS`：`repaint`（默认 `False`）、`repaint_denoise`（0.60）、
`repaint_steps`（8）。

降级策略是硬要求：**拿不到引擎就跳过，任务照常出最终图**。`probe()` 打不到 `/system_stats`
就写 `{"available": false, "reason": ...}` 加一条 warning；`RepaintError` 记
`{"available": true, "reason": ...}` 并回退磨皮结果。任务状态机里没有任何一条路径会因此失败。

掩膜检测在原图上做（`bgr`），重绘输入是磨皮结果（`out`）——反过来会让检测器被自己的柔化
误导。融合式 `region*(1-alpha) + repainted*alpha`，`alpha` 来自膨胀模糊后的掩膜，
所以掩膜外是纯拷贝，逐字节不变。

### 11.2 检测器是瓶颈，不是引擎

同一张注入基准、同一套参数、只换掩膜来源：

| 掩膜来源 | 覆盖注入痘点 | 磨皮后 resid | 重绘后 resid | 磨皮后 hf | 重绘后 hf |
| --- | --- | --- | --- | --- | --- |
| 自动检测（网页端默认） | 45.2% | 4.80 | **4.10** | 6.46 | **5.46** |
| 真值 discs（`--mask` 注入） | 100% | 6.20 | **4.53** | 6.57 | **4.44** |

结论明确：引擎本身还能再压一截，差的是掩膜准不准。`tests/probe_hit_check.py` 就是这个对照
（它把真值掩膜从 `apply(mask=...)` 这个后门喂进去）。

### 11.3 为什么原来的检测尺度不对

`skin_core.retouch_channel` 的 `mid` 半径是 `face_px * (0.030 + 0.035*blemish)`，
780 px 的脸约 42 px——那是**色斑尺度**，因为它要整片压平。而半径 3–8 px 的痘点被
光环晕开后有效宽度只有 10–16 px，在 42 px 带通里几乎不剩。

把带通缩到 `face_px * (0.014 + 0.020*blemish)`（约 22 px）后，同一 spill 预算下命中率
从 **20.3% 升到 29.6%**，网页端覆盖率从 24.3% 升到 45.2%，端到端残差改善从 0.02 变成 0.31。

扫描数据（`tests/probe_band_scan.py`，p95 阈值）：

| 中尺度半径 | 痘点命中 | 皮肤误检 |
| --- | --- | --- |
| 9 px | 25.6% | 4.75% |
| 14 px | 29.2% | 4.70% |
| 19 px | **29.6%** | 4.70% |
| 27 px | 28.0% | 4.72% |
| 43 px（原实现） | 20.3% | 4.81% |

顺带排除了两条死路：**top-hat 形态学**（k=25 时 25.7%，最好也只与窄带通持平）与
**磨皮改动量当掩膜**（`discs` 内 mean 3.08 vs 皮肤其余 2.47，几乎不可分，因为磨皮对
全皮肤都衰减 σ=3 颗粒）。两条都记在 `tests/probe_band_scan.py` / `probe_mask_diag.py` 里。

### 11.4 仍然开着的口子

自动掩膜在注入基准上只能覆盖 45%，而这个基准其实比真实更友好——注入的红斑是柔和的
midband 信号，与天然肤色不均本来就难分。真实痘点更锐利，理论上更好检，但手上两张
20 MP 真片都过于干净（第 5 节、第 9 节），测不出上限。**要往下推，缺的是一张真有
瑕疵的原片，不是更复杂的算法。**
