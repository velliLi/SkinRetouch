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
   在本机不存在。~~且机器无外网，装不上。~~ **该判断已被推翻**：网络其实是通的
   （`curl` 的 Schannel 伪失败误导了判断），MI-GAN 25.8 MB 与 LaMa 196 MB 都已下载并接入，
   见第 13 节。现在默认引擎是 MI-GAN，SDXL 降级为可选。

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

8. **默认引擎已从 SDXL 换成 MI-GAN，理由不是"更小更快"而是"不毁纹理"**（第 13 节）。
   用户报的"重绘会在皮肤上生成斑点"实测成因是采样式 inpaint **把掩膜内的高频洗掉了**：
   同一掩膜同一张图，`sdxl` 把掩膜内高频从 `4.15` 压到 `3.00`（−28%），擦除式的
   `migan`/`lama` 保持 `4.28`/`4.30`。代价从 `171 秒 / 6.9 GB / 需要 ComfyUI`
   降到 `3.1 秒 / 25.8 MB / 只要 torch`。

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

## 12. 引擎待命：从「要你先开 ComfyUI」到「服务自己管」

第 11 节的形态有个操作层面的硬伤：重绘需要一个跑着的 ComfyUI，而它不会自己出现。
一个勾选框配一句"需要 ComfyUI"，等于把启动、等待、关停三件事全丢给使用者。第 12 节
把这件事收进服务里。

### 12.1 三条路径，优先级明确

`server/engine.py` 的 `Engine` 只有一条判断链，越靠前越优先：

1. **`repaint_url` 上已有实例在应答** → 直接复用。`managed` 保持 `False`，`stop()` 什么都不做，
   页面按钮显示「占用中」。这是别人的进程，任何情况下都不能接管或关闭。
2. **自己 spawn 过且进程还活着** → `managed=True`，按钮变「停止」，可以正常关。
3. **都没有** → `idle`。`--repaint-autostart`（默认开）时启动阶段就走到这里 spawn；
   关闭时留在 `idle`，等页面按钮或 API 触发。

判据是 `repaint.probe()`（GET `/system_stats`），和真正跑重绘时用的是同一个探活函数，
不存在"探活通过但推理连不上"的错位。

### 12.2 实测时间

| 动作 | 耗时 | 说明 |
|---|---|---|
| 冷启动到 `ready` | **27.2 s** | spawn → 加载 6.9 GB checkpoint → 应答 `/system_stats` |
| `stop` → `idle` | < 2 s | terminate + 等 15 s 上限 |
| `stop` → `start` → `ready` | **22.6 s** | 进程还热，比首次快 |
| `--no-repaint-autostart` 下点「启动」 | **38.0 s** | 服务本身没预热，冷一些 |
| 单块重绘（引擎已待命） | 12–24 s | 首块 32.7 s，含把 checkpoint 搬进 VRAM |
| 整个 job（2 块 + 磨皮） | 64.6 s | `pipeline_ms 7440` + `repaint_ms 56349` |

最后两行值得放在一起看：**待命时间是一次性的，重绘时间才是每张图的成本。** 27 秒的
代价换来的是后面每张图省掉冷启动那句"等 30 秒"，这个交易只有把服务开一整天时成立。

### 12.3 唯一需要小心的地方

`stop()` 必须只杀自己 spawn 的。`_proc` 为空或已退出时 `managed` 为 `False`，此时
`stop()` 直接返回当前状态，不发任何信号。这条约束在 `tests/server_test.py` 里被钉住
（`engine is never managed until we start one`、`stop answers without touching a foreign
engine`），因为误杀用户自己开着的 ComfyUI 是不可接受的副作用，比功能缺失严重得多。

---

## 13. 引擎换代：MI-GAN / LaMa 与高频回贴

起因是一句用户反馈："目前重绘模型效果并不好，会在皮肤表面生成斑点；尝试寻找更加轻量化、速度更快的重绘模型。"
这一节记录三件事：症状的真实成因、三个引擎的对照数据、以及为什么最后落地成
「擦除式模型 + 从原图搬高频」。

### 13.1 症状不是"模型画了脏东西"，是"模型把干净的纹理洗掉了"

在注入式夹具上量 `spk_rat`（掩膜内中位 Laplacian 除以掩膜外环带的中位 Laplacian，
大于 1 表示掩膜里比周围更"糙"，也就是造了新斑）：SDXL 是 `0.667`，**小于 1**。
两张 20 MP 真实照片上分别是 `0.750` 和 `—`（偏低）。所以在像素统计意义上，
SDXL 并没有往皮肤上加斑点，它做的是反向的事：**把掩膜内的纹理抹平**。

决定性的一行数据来自 `tests/repaint_test.py`——同一个测试图、同一个掩膜、同一段磨皮输出：

| 引擎 | 掩膜内平均改动 | 掩膜内高频能量 |
|---|---|---|
| 输入（磨皮后） | — | 4.15 |
| `migan` | 7.27 | **4.28** |
| `lama` | 5.19 | **4.30** |
| `sdxl` | 6.99 | **3.00** |

SDXL 把掩膜内高频压掉 28%，掩膜里只剩下它自己从提示词方向"编"出来的低频起伏。
一块没有毛孔、没有细纹、只有缓慢明暗的圆斑贴在真实皮肤上，人眼读到的就是"起了一层斑"。
这解释了为什么把 `denoise` 从 0.30 调到 0.60 都没有改善——问题不在幅度，在于采样式
inpaint 的语义就是"这块重画"，而重画必然丢掉它没见过的原有纹理。

### 13.2 注入夹具上的完整对照

`tests/probe_small_inpaint.py`，t01 素材，896×1152，掩膜 1.654%（4094 px），环带 59572 px，
皮肤覆盖 0.3227。`tex` = 掩膜内高频 / 环带高频，`spk_rat` 同上，`ms` 是单张耗时（CPU）。

| state | hf | resid | ssim | spk_rat | tex | ms |
|---|---|---|---|---|---|---|
| reference（干净原图） | 5.325 | 0 | 1.0000 | 1.000 | 0.904 | — |
| blemished（注入瑕疵） | 8.910 | 7.594 | 0.9539 | 0.875 | 0.936 | — |
| retouch（CPU 管线） | 7.015 | 6.136 | 0.9569 | 1.000 | 0.932 | 520 |
| retouch + sdxl（d0.6 s8） | 5.324 | 4.540 | 0.9557 | 0.667 | 0.707 | **171046** |
| retouch + migan（单块 512，错） | 2.205 | 5.191 | 0.9556 | 0.167 | 0.293 | 922 |
| retouch + migan（1:1 平铺） | 3.399 | 4.549 | 0.9555 | 0.333 | 0.452 | 3095 |
| migan + hf 1.0 | 5.883 | 4.390 | 0.9569 | 0.833 | 0.777 | 3095 |
| **migan + hf 1.1** | 6.665 | 4.487 | **0.9569** | 0.833 | 0.881 | **3095** |
| migan + hf 1.3 | 7.494 | 4.616 | 0.9568 | 1.000 | 0.990 | 3095 |
| **retouch + lama** | 5.746 | **4.349** | 0.9556 | 0.833 | 0.765 | 8458 |
| lama + hf 1.15 | 6.755 | 4.413 | 0.9569 | 0.833 | 0.892 | 8458 |

读法：

- **`resid` 三者几乎一样**（4.35–4.55）。都会把瑕疵抹掉，这不构成选择依据。
- **`tex` 才是分水岭**：`sdxl 0.707` 对 `migan+hf 0.881` / `lama+hf 0.892`。干净原图是 0.904。
- **耗时差 55 倍**，权重差 267 倍。
- `spk_rat` 全表没有超过 1.0 的，说明"加重瑕疵感"不是这些引擎在小掩膜上的失效模式；
  失效模式是"糊"，即 `tex` 掉下来。

`migan` 那行标了"错"的是踩过的坑：MI-GAN 的 traced 图输入被固定成 512×512，
把人脸整块 crop 缩进一个 512 窗口后，细节在缩放里就没了，`tex` 只有 0.293。
必须**按 1:1 尺度在 512 窗口上平铺**，只对窗口内的掩膜做前处理。

### 13.3 高频回贴：把毛孔从原图搬回来，而不是让模型合成

擦除式模型（MI-GAN / LaMa）的输出语义是"这块本来什么都没有"，所以给的是干净、平滑、
无纹理的低频填补。而瑕疵的特征是局部的、低频的色偏和隆起——**高频是好的**。
于是 `graft_hf(base, texture_src, mask, sigma=2.5, strength)`：

```
high = texture_src − GaussianBlur(texture_src, sigma=2.5)
out  = base + high × (mask/255) × strength
```

掩膜 alpha 做权重，掩膜外严格不碰（落地时再加一道 `keep = mask == 0; merged[keep] = out[keep]`
兜底，`tests/repaint_test.py` 断言 `changed_outside == 0`）。`strength` 就是参数 `repaint_hf`：

夹具上扫过的点（migan，`tex` / `spk_rat`）：`0 → 0.452/0.333`、`1.0 → 0.777/0.833`、
`1.1 → 0.881/0.833`、`1.3 → 0.990/1.000`。lama 只测了单点 `1.15 → 0.892/0.833`
（不回贴时是 `0.765`）。干净原图是 `0.904`。

`1.3` 已经过冲到 0.990（高于干净原图），`spk_rat` 顶到 1.000，等于自己往皮肤上加纹理，
上限就定在 1.2。默认值 `1.15` 取自真实照片那一组（`tests/probe_real_small_inpaint.py`
用的就是这个值，两张图落在 0.83–1.01），不是夹具上的最优点——夹具是合成瑕疵，密度比
真实照片高，会偏保守。

### 13.4 真实照片：引擎之间的差距被掩膜尺度掩盖了

`tests/probe_real_small_inpaint.py`，两张 20 MP 真实人像，掩膜由生产路径的
`blemish_mask` 自动检测（2.05% / 2.23%），表里只列关键量。

| 素材 | retouch | migan | migan+hf1.15 | lama | lama+hf1.15 | sdxl |
|---|---|---|---|---|---|---|
| DSC_4081 `tex` | 0.824 | 0.520 | 0.830 | 0.869 | 0.835 | 0.843 |
| DSC_7300 `tex` | 1.006 | 0.621 | 0.991 | 0.980 | 1.008 | 0.942 |
| DSC_4081 耗时 | — | 3.1 s | 3.1 s | 6.5 s | 6.5 s | **116.3 s** |
| DSC_7300 耗时 | — | 3.1 s | 3.1 s | 6.4 s | 6.4 s | **116.4 s** |

结论要说清楚：**在真实照片的弱掩膜上，`sdxl` 的 `tex` 并不难看**（0.843 / 0.942，和 `retouch`
的 0.824 / 1.006 一个量级）。差距是在测试夹具那种"密集、成串、半径 3–8 px 的痘点"上才
拉开的——那时候 SDXL 要连续重画很多小圆块，每块都丢纹理，平均下来就是 −28%。
所以换引擎的第一收益是**速度与部署成本**（116 秒 → 3.1 秒；6.9 GB + ComfyUI → 25.8 MB + torch），
第二收益才是纹理保真，且第二收益随瑕疵密度上升。

### 13.5 别在大掩膜上试这三个引擎

`tests/probe_wide_mask.py`，掩膜铺到 33.1% 皮肤（`grow=2, blur=3.0`），t01 分两块：

| state | hf | tex | 块间均值 spread | 缝处梯度 | 缝旁梯度 | ms |
|---|---|---|---|---|---|---|
| retouch | 4.829 | 0.927 | 8.708 | 1.042 | 1.052 | — |
| migan | 5.647 | 1.083 | 7.281 | 1.062 | 1.076 | 3563 |
| lama | 5.585 | 1.072 | **4.180** | 1.067 | 1.083 | 7769 |
| sdxl | 4.407 | **0.845** | 9.375 | 1.088 | 1.102 | **363549** |

块均值（BGR）：`retouch [(110.7,131.2,163.0), (89.3,114.6,148.8)]`、
`migan [(77.5,100.3,129.1), (88.9,114.3,147.5)]`、`lama [(84.2,106.5,135.9), (89.0,114.1,148.6)]`、
`sdxl [(111.5,132.1,163.2), (88.3,114.2,148.1)]`。

`migan`/`lama` 把上面那块压暗了 33 / 26 灰阶——它们把光照梯度当成"要抹掉的低频"一起抹了，
于是块与块之间出现明显色阶台阶（spread 掉到 7.3 / 4.2）。`sdxl` 保住了块间色调（111.5 对 110.7），
但把纹理压到 0.845 并耗时 363 秒。**没有引擎适合大掩膜**，生产路径的 `blemish_mask`
只产 1.6–2.2% 的稀疏掩膜，就是这个原因。

### 13.6 权重从哪来：网络判断被彻底更正

之前这一节写的是"机器无外网，装不上"。**那是错的**，错在测量手段：`curl` 在这台机器上
走 Schannel，对 `https://github.com` 报 `000` 超时，看起来像全断。改用 python urllib 实测：

| 站点 | 结果 |
|---|---|
| `https://github.com` | 200 / 0.38 s |
| `https://api.github.com` | 200 |
| `https://hf-mirror.com` | 200 / 2.74 s |
| `https://www.modelscope.cn` | 200 / 0.53 s |
| `https://raw.githubusercontent.com` | 200（约 30 s，慢） |
| `https://huggingface.co` | 超时（唯一不通的） |

权重本体在 GitHub Releases（`Sanster/models`），不在 HF，所以下载没有被卡住：

| 模型 | 文件 | 体积 | MD5 |
|---|---|---|---|
| MI-GAN | `migan_traced.pt` | 27,048,675 B | `76eb3b1a71c400ee3290524f7a11b89c` |
| LaMa | `big-lama.pt` | 205,669,692 B | `e3aa4aaa15225a33ec84f9f4bc47e500` |

两者都是 TorchScript，`torch.jit.load` 直接跑，不需要 diffusers / transformers / ComfyUI。
下载器 `tests/fetch_repaint_weights.py` 踩过两个坑，都值得记下：

- **按文件均分 16 段（每段 12.8 MB）会让 GitHub 代理直接卡死**：零字节卡 82 秒。
  改成 2 MB 小片 + 共享队列抢任务后，吞吐从 7–11 KB/s 拉到 800–990 KB/s。
- **196 MB 在 60.2% / 73.4% 处超时后从头重来**，浪费两次。加 `<dest>.progress` 记录
  已完成切片索引后变成断点续传，最终 MI-GAN 32 秒下完、LaMa 三次续传下完。

镜像实测（512 KB Range）：`edgeone.gh-proxy.com` 40–110 KB/s、`gh-proxy.com` 76–103 KB/s、
`ghproxy.net` 8–9 KB/s、github 直连 7–11 KB/s、`ghfast.top` DNS 解析失败。
脚本的做法是下载前对每个镜像测速择优，再开并发拉片。

### 13.7 网页端全链路实测

`tests/probe_web_repaint.py --repaint-engine {migan,lama}`，走真实 HTTP（`/api/jobs`），
同一次提交跑两遍（`repaint=0` / `repaint=1`），自动掩膜覆盖注入面积的 45.2%：

| 状态 | 瑕疵区 `resid` | 掩膜覆盖区 `resid_hit` | 瑕疵区 `hf` | 重绘耗时 |
|---|---|---|---|---|
| blemished | 7.59 | 6.56 | 8.24 | — |
| retouch | 6.14 | 4.80 | 6.46 | — |
| + `migan` | **5.79** | **4.02** | 6.55 | 4.08 s |
| + `lama` | 5.83 | 4.12 | 6.52 | 8.79 s |
| + `sdxl`（第 11 节旧测） | 5.83 | 4.10 | **5.46** | 22.6 s |

`sdxl` 那一列的 `hf 5.46` 是本章论点的独立复现：它把瑕疵区的高频从 6.46 压到 5.46，
而擦除式两个引擎停在 6.5。整图 `hf` 略微上升不是退化——回贴的目的是让掩膜内恢复到
"有毛孔"的状态（干净原图 5.03，注入的 σ=3 颗粒把坏图推到 8.24），停在 6.5 就是
"和周围一样的真实质感"。

### 13.8 落地形态与门禁

`server/repaint.py` 现在是三引擎共存：`ENGINES = ("migan", "lama", "sdxl")`，`DEFAULT_ENGINE = "migan"`。
`migan`/`lama` 走 `_apply_nets`（纯 CPU，单次调用，无分块），`sdxl` 走原来的 `_apply_tiles`。
参数从 13 个变成 15 个，新增 `repaint_engine` 与 `repaint_hf`，`repaint_denoise`/`repaint_steps`
的说明改成"仅 sdxl 引擎使用"并在选另两个引擎时于页面上灰掉。

门禁（全绿）：`blemish_test BASELINE OK (3 cases)`、`node_test NODE OK`、
`compare_paths PATHS OK (rel 1.44%, p99 13.0)`、`server_test SERVER OK 61 checks`、
`service_test SERVICE OK 37 checks`、`repaint_test REPAINT OK 30 checks`（三个引擎各 7 项 + 跳过项）。
