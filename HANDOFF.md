# 交接文档

写给接手这个目录的下一次对话。假设你对本项目一无所知，读完这一份就能直接干活，不必重新摸索。

---

## 这是什么

一个只做美肤的轻量人像修图管线。输入一张照片，输出同一张照片，皮肤上的痘印、色斑、泛红被去掉，毛孔、睫毛、唇纹、鼻翼边缘保持原样。

它处理**主要人物的脸，以及露出来的身体皮肤**——脖子、肩、手臂、手、胸口。20.7 MP（5568×3712 或 3712×5568）的竖幅或横幅全身照是它的目标场景。

它不是生成模型。全程只有两个神经网络，职责都只是**限定搜索范围**：YOLOv8-face（50 MB）框出人脸，YOLOv8-seg（55 MB）框出主人物。剩下全是确定性的 OpenCV / torch 数学运算，同一张图跑一百次结果完全一致。

当前状态：可用，已通过注入瑕疵的对照实验验证。既是 ComfyUI 插件，也能脱离 ComfyUI 单独调用。

---

## 三十秒跑通

```powershell
$env:TMP="E:\comfyui\ComfyUI\temp"; $env:TEMP="E:\comfyui\ComfyUI\temp"
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\tests\blemish_test.py
```

打印三行 `hf-energy`，然后往 `tests/output/` 里写三张 `*_compare.png`。看到 `DONE` 就是好的。

数值应当与下面完全一致（本项目任何改动都以此为回归基线）：

```
t01 hf-energy clean=5.9 blur=1.3 ours=7.5 coverage=0.320
t07 hf-energy clean=6.1 blur=2.0 ours=9.4 coverage=0.264
t05 hf-energy clean=7.4 blur=2.0 ours=8.3 coverage=0.011
```

节点契约：

```powershell
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\tests\node_test.py   # 期望 NODE OK
```

成品图与分档计时（独立脚本，不依赖前一步）：

```powershell
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\tests\deliver_demo.py  # 写 deliver_pair / deliver_zoom
```

图形界面 / 远程接口（两条自检，互相独立）：

```powershell
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\tests\server_test.py    # 期望 SERVER OK 47 checks
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\tests\service_test.py   # 期望 SERVICE OK 37 checks
```

日常使用不必碰命令行：双击 `E:\SkinRetouch\服务开关.bat`，没跑就起（并打开浏览器），在跑就停。细节见下方「一键开关」。

### 关于输出目录

测试脚本默认写 `tests/output/`。想换地方就设 `$env:SKIN_OUT`。

`$env:TMP` 必须指向一个可写目录，否则 ultralytics 与临时文件会报 `FileNotFoundError: No usable temporary directory found`。

20 MP 端到端（会写几张 30 MB 级的 PNG）：

```powershell
$env:SKIN_DEVICE="cuda"   # 或 cpu
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\tests\probe_body_final.py
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\tests\dump_compare.py   # 1:1 对比
```

---

## 环境事实

| 项 | 值 |
|---|---|
| 解释器 | `E:\comfyui\python_embeded\python.exe`（Python 3.11.9） |
| 显卡 | RTX 3060 Laptop 6 GB，ComfyUI 常驻时约剩 5 GB |
| OpenCV | `cv2 4.10.0`，带 contrib（`ximgproc.guidedFilter` 可用），**纯 CPU 构建**，`cv2.cuda` 无设备 |
| 检测器 | `ultralytics 8.3.40` |
| torch | `2.5.1+cu121`，CUDA 可用 |
| 模型 | `models/face_yolov8m.pt` 50 MB、`models/person_yolov8m-seg.pt` 55 MB，随项目自带 |
| ComfyUI 挂载 | `E:\comfyui\ComfyUI\custom_nodes\ComfyUI-SkinRetouch` 是指向本目录的 Junction |

不新增任何依赖，不需要联网。`include_body=False` 时完全不碰显卡。

关于 `device="cpu"`：`detect_faces()` 里写死了 CPU。换 GPU 每个像素只省 4 ms——20 MP 的瓶颈是 letterbox 预处理的 `cv2.resize`，推理本身在 640×640 上几十毫秒就完了，实测 CPU 186 ms / CUDA 190 ms，一样。真要加速得先缩图再检，那会改框坐标。

---

## 不要推翻的决定

这几条是踩过坑、量过数据之后定下来的。想改之前先读理由。

**1. 主路径不用扩散模型重绘；重绘只作为默认关闭的可选项，且只在小掩膜上。**
美肤是信号处理问题，不是重绘问题，主路径必须保持 `skin_core` 的频域保边滤波。但"完全不能用重绘"这个判断已被实测修正（全文见 `REPAINT_EXPLORE.md`）：

- **大掩膜确实会毁图**，原来的担忧在这个场景成立：把掩膜铺到整张皮肤（覆盖 39.5%）后，SSIM 从 0.954 掉到 0.899，瑕疵区残差反而从 7.95 恶化到 8.39，分块拼接还留下 4.66 的接缝梯度跳变（相邻行只有 2.2–3.0）。
- **小掩膜不会**：稀疏痘点掩膜（覆盖 1.4%）分块重绘后，接缝梯度与坏图逐位相同（3.6150 vs 3.6172），掩膜外像素逐字节不变（`tests/repaint_test.py` 断言 `changed_outside == 0`），发丝一根没断。
- **收益有据**：先用 `skin_retouch` 磨皮、再补一过重绘，瑕疵区残差 6.20 → 4.53、高频能量 6.57 → 4.44（真值掩膜，`tests/probe_hit_check.py`）。

落地形态：网页服务的 `repaint` 开关，默认 `False`，只在 CPU 磨皮之后追加一过，掩膜由 `server/repaint.py` 自动检测（红通道窄带通 + 分位阈值）。**ComfyUI 不在线就跳过并写警告，绝不让任务失败。**
**目前最大的空白：人脸身份漂移没有量化。** 本机 `~/.insightface/models` 为空（无 ArcFace 权重）、也没有 `lpips`/`pyiqa`，只有"掩膜外漂移 4.06 ≈ 纯 VAE 往返误差"这个代理证据，和 5 倍放大图的肉眼判断。要下"重绘不会改身份"的结论，得先补测量手段。

**2. 不用 mediapipe / face mesh。**
本机 `mediapipe 0.10.14` 配 `protobuf 5.29.5` 初始化即崩溃，`FaceDetection` 与 `FaceMesh` 都一样：

```
[libprotobuf ERROR text_format.cc:335] Error parsing text-format
mediapipe.CalculatorGraphConfig: 15:22: Expected identifier
RuntimeError: Failed to parse
```

降级 protobuf 会打断 `onnx 1.20`，升级 mediapipe 要动 `python_embeded`，两个代价都比收益大。改用「YOLO 框 + 椭圆先验 + 肤色判定」硬算，效果够用且零额外依赖。

**3. 检测不用 GPU，滤波必须用 GPU。**
检测器每次推理 0.19 秒，且 CPU 与 CUDA 实测同为 0.19 秒——20 MP 的预处理 `cv2.resize` 是瓶颈，不是卷积。但**滤波必须上卡**：`cv2.ximgproc.guidedFilter` 没有 CUDA 后端，本机 OpenCV 也是纯 CPU 构建（`cv2.cuda.getCudaEnabledDeviceCount()` 返回 0）。20 MP 上 r=66 的导向滤波 cv2 要 410 ms，torch 版 61 ms。全身美肤要整幅人物区域做全分辨率滤波，CPU 路径单张 9.8 秒，GPU 路径 3.6 秒。

`skin_gpu.py` 是自己写的 He 快速引导滤波。**踩过一次精度坑**：`_box_mean` 最初用二维二次 cumsum，20 MP 上中间值累积到 `255*W*H ≈ 5.3e9`，float32 只有 7 位有效数字，窗口差分损失数个单位，于是导向滤波在平滑手臂上印出一条 Z 形等值线。修法是分离式一维 cumsum，每步差分后立刻除以核长，中间值最大只有 `255*H ≈ 1.4e6`。修完全图与 cv2 的 `max|d|` 降到 1.4–3.6 灰阶，Z 线消失。

**4. 逐脸 ROI 处理，不是全图滤波。**
最初是全图滤波，5568×3712 要 21.4 秒，且 `r_mid` 撞到 64 px 上限，对 3487 px 的脸来说去痘尺度只有脸宽的 1/54。改成每张脸只在自己的 ROI（框外扩 30%）内计算，降到 5.0 秒。**算法零改动，验收标准是像素级一致**——改造后 `hf-energy` 与 `coverage` 逐位相同，说明只是缩小了计算域。

**5. 羽化半径随脸大小缩放，不是固定像素。**
固定 8 px 会毁掉全身照里的小脸：170 px 的脸上羽化 8 px，掩膜被啃到只剩 0.6%。现在按 `face_px * (0.004 + 0.020 * feather)` 走。

---

## 算法怎么走

每一张脸独立过一遍：

**掩膜**（`face_skin_mask`）三层相乘：

1. 脸型椭圆先验 `0.505 × 0.60` 半轴，中心下移 `0.03`，额头在 `cy - fh*0.38` 以上切掉（YOLO 框会含进头发）。
2. 自适应 YCrCb 色距。参考色取**四个取样块合并后的中位数**——左右脸颊 `(±0.20, 0.14)`、鼻梁 `(0, 0)`、下巴 `(0, 0.22)`，每块是半个脸宽 11% × 半脸高 7.5% 的矩形，四个块的像素摊平后一起取中位数（不是四个数再取中位数）。单点采样会落到鼻孔阴影或眼睛上，全身照因此整张脸判空。
3. 结构抑制：亮度低于 `0.26 × ref_y` 的（眼、眉、睫毛、鼻孔影）压掉；色相 `(hue<22 或 hue>330) 且 sat>0.52` 的（唇红）乘 0.18。

色距计算里 **Y 的权重是 0.35**，不是 1.0。这是关键：逆光脸上亮度从 90 到 220 起伏，用等权欧氏距离会把半个脸当非皮肤，实测 `dist` 中位数飙到 50。

**滤波**（`retouch_channel`，在 CIELab 的 L 上做）：

```
fine    = guided(I,    r_fine)     # 去毛孔以下的传感器噪声
mid     = guided(fine, r_mid)      # 3–15 px 痘印消失，得到干净基底
midband = fine - mid               # 8–30 px，正是痘与斑的尺度 → 丢弃
highband= I - fine                 # 毛孔尺度 → 保留
out     = mid + midband*0.12*keep + highband*(1 - kill*w_edge)
```

`w_edge = exp(-(highband/tau)²)`，`tau = 2.5 × MAD(highband)`。**这一项是它不像塑料脸的根本原因**：平坦区域的 `highband` 小，`w_edge≈1`，噪点被 `kill` 碾掉；睫毛、唇线、鼻孔边缘的 `highband` 大，`w_edge≈0`，结构原样留下。

曾经写错过一次：把 `midband` 当纹理保留，结果痘印全留下来了——因为中频带恰好就是痘的尺度。现在的 `0.12*keep` 系数就是那次留下的。

**色度**（`tone_even`）：a/b 通道向半径 `face_px*(0.08+0.10*tone_even)` 的局部均值收敛，`t = sqrt(tone_even)*strength*weight`。色斑本质是色度缺陷，不是亮度缺陷，在 L 上做再多也没用。

**收尾**：掩膜内平均亮度对齐回原图，防止整体提亮或发灰。多张脸串行处理，每轮结果都 clamp 回 0–255，否则浮点残留会被喂进下一次 `cvtColor`。

---

## 身体怎么走

`include_body=True`（默认）时，人脸处理完再走一遍全身。三件事，顺序不能换：

**1. 主人物掩膜**（`skin_body.detect_primary_person`）。`person_yolov8m-seg` 出分割，**取面积最大的那一个**——`scores` 只用于回报，不参与选择（`skin_body.py` 里 `idx = int(np.argmax(areas))`）。`imgsz=640` 是实测选出来的——1280 上置信度反而掉到 0.42~0.73 并碎片化。

ultralytics 返回的 masks 在 letterbox 网络空间（`[n,640,448]` 这种），不是原图坐标。别用它内部的 `uops.scale_masks`（传 numpy 会抛 `'numpy.ndarray' object has no attribute 'dim'`），自己反算即可：`gain = min(mh/h, mw/w)`，pad 居中，裁出来再 `cv2.resize` 回原幅。

**2. 肤色软掩膜**（`skin_body.body_skin_mask` / `skin_gpu.body_skin_mask_gpu`）。这是全项目最难的一段，判据是三层相乘：

- **色度椭圆**，参考色**必须锚定人脸**（四个取样块像素合并取中位数）。在 person 掩膜内做迭代中位数估参考色会收敛到全图中位数 `[140,134,128]`——固定点陷阱，一次就不再变，整件衣服全被选中（实测占 person 面积 80%）。半径 `cr_tol=9.0 / cb_tol=7.0`。
- **高光补偿**：Cb/Cr 差异化。过曝皮肤 `dCb` 只偏 +5，浅色赛车服偏 +12 以上，所以 Cr 放开 3.2 倍、Cb 只放开 1.6 倍。Cb 一起放开会把衣服拉回来。
- **方向性纹理抑制**（`texture_suppression`）：织物有单一主导梯度方向，`aniso = (Ex-Ey)/(Ex+Ey)` 在 9 px 窗口上显著为正，皮肤接近 0。实测皮肤 aniso −0.08~+0.05，织物 +0.32~+0.41。**必须在原生分辨率算**，降采样（`INTER_AREA`）会把 2–3 px 的织物条纹抹平，判据直接失效。
- 最后乘 person 掩膜（`> 0.75`，不是 0.5——0.5 会把上采样模糊边界放进来），收缩 9 px，开 11 / 闭 31，填掉小于 person 面积 2% 的内部空洞（`hole_below=160`，把羽化裙边一并算作洞）。

**3. 滤波**。尺度 `body_scale_px = max(face_px, 0.20*短边)`，和 `retouch_channel` 同一套。GPU 上一次上卡全图算完；CPU 上按 640 px 横条切（`_retouch_body_strips`，pad 取 `max(192, r_mid+16)`）。

**脸不能再被磨一次**。body 掩膜在脸部会被 `_face_ellipse_mask` 挖掉（`soft_b *= 1 - fcut`），否则同一块皮肤被 flat + body 各磨一遍，读起来像塑料。

**已知不可分**：浅粉/肉色紧身衣的某些区域（`Cr147 Cb123`）与皮肤的色度**完全相同**，DSC_7300 左腰那片就是。任何纯色彩判据都分不开，纹理判据也救不了（那块布料本身平滑）。接受它——那块布料平滑意味着美肤对它的实际改动接近零，1:1 对比肉眼无法分辨。

---

## 旋钮

节点 `Skin Retouch (Lite)`，分类 `image/retouch`，输入 `IMAGE` 输出 `(IMAGE, MASK)`。

| 参数 | 默认 | 作用 |
|---|---|---|
| `strength` | 0.75 | 总强度 |
| `texture_keep` | 0.35 | 保留的毛孔纹理。0 = 塑料，0.6 = 几乎没动 |
| `blemish` | 0.7 | 去痘尺度，连带决定 `r_mid` |
| `tone_even` | 0.5 | 泛红/色斑均匀化 |
| `feather` | 0.5 | 掩膜边缘软度，相对脸大小 |
| `max_faces` | 4 | 处理几张脸 |
| `face_confidence` | 0.35 | 检测阈值 |
| `include_body` | True | 是否连身体皮肤一起处理，关掉就只剩脸 |
| `body_strength` | 0.0 | 身体强度。0 = 跟随 `strength` |
| `device` | `auto` | `auto` / `cuda` / `cpu`。CUDA 不可用时自动降级 |

**注意 `include_body` 有两个默认值**：节点上默认 `True`，而 `skin_core.skin_retouch()` 的函数默认是 `False`。这是刻意的——库级默认保持向后兼容，老代码升级后行为逐像素不变；节点面向使用，默认就该把身体带上。直接 import 调用时若想处理身体，必须显式传 `include_body=True`。

**`device` 同样有两个默认值**：节点 `"auto"`，函数 `None`（等价于 `"cpu"`）。所以 `tests/blemish_test.py` 里那三条 hf-energy 基线**是 CPU 路径的产物**，GPU 路径不在它们的约束范围内。两条路径的差距由 `tests/compare_paths.py` 单独守着（当前 rel 1.44%、p99 13 灰阶）。

预设：轻度 `0.5 / 0.55 / 0.5`，标准 `0.75 / 0.35 / 0.7`，商业 `0.9 / 0.18 / 0.85 / tone_even 0.7`。全身照建议 `body_strength` 比 `strength` 低 0.1~0.2——身体的毛孔本来比脸粗，同强度会显得发虚。

代码入口：

```python
import skin_core
out, mask, info = skin_core.skin_retouch(bgr, strength=0.75, texture_keep=0.35,
                                         blemish=0.7, tone_even=0.5, feather=0.5,
                                         conf=0.35, max_faces=4, model_path=None,
                                         include_body=True, body_strength=None,
                                         device="auto")
```

`info` 是每张脸的 `{box, face_px, feather_px, roi, coverage}`；开身体后多一条 `kind="body"`，带 `{box, scale_px, ref, coverage, backend, ms}`。`coverage` 是**该 ROI 内** `mask>127` 的占比，不要和全图占比混用——`blemish_test` 里打印的那个是 `(m2>180).mean()` 全图口径，两者数值不同是正常的。

---

## 实测数据

CPU，含检测的稳态耗时。耗时跟着**人脸像素尺寸**走，不跟画幅走：

| 图幅 | 人脸 | 耗时 |
|---|---|---|
| 448×576 | 313 px | 274 ms |
| 896×1152 | 778 px | 457 ms |
| 1344×1728 | 1167 px | 743 ms |
| 2784×1856 | 1743 px | 1.44 s |
| 5568×3712（20 MP 半身） | 1538 px | 1.47 s |
| 5568×3712（20 MP 满幅特写） | 3487 px | 5.0 s |

20 MP 半身照和 5 MP 半身照都是 1.4 秒上下，多出来的背景不参与计算。

身体路径是另一回事：它必须在整幅人物上算，无法缩小计算域。两张示例图（20.7 MP）热态实测：

| 阶段 | DSC_7300（竖 3712×5568） | DSC_4081（横 5568×3712） |
|---|---|---|
| 人脸检测 + 逐脸处理 | 238 ms | 205 ms |
| 人物分割 | 694 ms | 940 ms |
| 身体掩膜 | 742 ms | 797 ms |
| 身体滤波 | 609 ms | 579 ms |
| 羽化、合成、均值对齐 | 1371 ms | 1767 ms |
| **合计（GPU）** | **3.65 s** | **4.29 s** |

同样是全身照，这两张的合计差了 0.6 秒，因为人物掩膜在 DSC_4081 上更碎（背景里还有八个人），分割和羽化都更贵。

**冷启动**：全新进程第一次调用要多付 20 秒左右——两个模型从磁盘加载 7 s，其余是一次性的 CUDA 上下文与算子初始化。这 20 秒只付一次，之后常驻。接进 ComfyUI 时第一次执行会明显卡住，不是死循环。

纯 CPU 跑同一套身体路径是 9.8 秒。纯脸路径（`include_body=False`）在 20 MP 上约 1.1~1.5 秒，具体跟人脸像素尺寸走。

人物掩膜覆盖率：DSC_7300 占画面 53.8%，DSC_4081 占 39.5%。身体皮肤掩膜占画面 13.2% / 18.0%，占人物面积 24.2% / 45.2%。

**验证方法**：生成模型出的图本身已被修过，拿干净对比图证明不了什么。所以 `blemish_test.py` 先在皮肤掩膜内注入 26 个痘点、10 片低频红斑、σ=3 高斯颗粒，再让管线和「暴力高斯模糊」各修一遍，和干净原图三方对比。高频能量表见开头，`ours / clean` 稳定在 1.1–1.5 之间，而模糊基线掉到 0.2–0.3——那就是塑料感的量化形态。

图在 `tests/output/`：`t01_compare.png`、`t07_compare.png`、`t05_compare.png` 各四栏（注入 / 模糊 / 本管线 / 干净），`deliver_zoom.png` 是细节特写。

---

## 已知限制

- **身体只处理"主人物"**。多人合影里第二个人之后的身体不碰（脸仍然处理，`max_faces` 管）。要多人身体得上逐人分割 + 逐个滤波，耗时线性增长。
- **浅色紧身衣会和皮肤混**。色度完全一致时（见上面「已知不可分」）无法分开，实际影响接近零。
- 人物分割在 640 上做，边界精度约 ±3 px 原图像素；靠 `edge_shrink=9` + 羽化吃掉。
- 小脸（<150 px）掩膜覆盖率明显下降，t05 那张 168 px 的脸只有 0.011。图还看得过去，但它本来也不值得磨。
- 侧脸、遮挡、口罩会降低掩膜质量。
- 极深肤色、纯色氛围灯（比如整场红光照脸）会让自适应色距失准，目前没暴露色距阈值参数，只能靠 `face_confidence` 间接干预。
- 不做美白和改色。要提亮接色彩节点。

---

## 图形界面与远程接口

`run_server.py` + `server/` 是一套零依赖的本地服务：同一个进程既提供网页界面，也提供 HTTP 接口。启动：

```powershell
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\run_server.py           # 127.0.0.1:8791
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\run_server.py --lan --open
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\tests\server_test.py    # 期望 SERVER OK 47 checks
```

`server_test.py` 自己在随机端口起一个实例，跑完断言再关掉，不碰 `server/data/`。

### 不要推翻的决定

1. **不用 Flask / FastAPI**。项目硬约束是不新增依赖，checkout 现在只要 cv2 + torch 就能跑，这一点比路由写起来舒服重要。全部基于 `http.server.ThreadingHTTPServer`。
2. **不做 Tkinter 桌面窗口**。Web 单页同时充当本机和远程界面，一套代码两用；Tk 出了这台机器就是废的。
3. **前端用 350ms 轮询而不是 SSE**。任务只要 3–10 秒，轮询代码量是 SSE 的三分之一，且不受 `http.server` 在 keep-alive 下处理无 `Content-Length` 流的限制。SSE 端点仍然提供，给脚本用。
4. **单 worker 串行队列**。6 GB 显存加上 ComfyUI 常驻，两张大图并跑必然 OOM。队列上限 8，满了返回 503。
5. **不改 `skin_core.py`**。进度只能粗粒度（decode → pipeline → encode），细分耗时从返回的 `info` 里回填，而不是往算法里插回调。
6. **参数走 query string，不走 JSON body**。这样浏览器可以直接用 `fetch` + 原始文件字节提交，和 `curl --data-binary` 是同一条路径。三种上传体（raw / multipart / base64 JSON）都在 `api.py:_extract_upload` 里。

### 参数名映射集中在一处

节点面向 `face_confidence` / `body_strength=0.0 表示跟随 strength`，而 `skin_core.skin_retouch()` 收 `conf` / `body_strength=None`。转换只在 `server/jobs.py:core_kwargs()` 里做一次，别在别处重复。

`server/config.py:PARAM_SPECS` 是 13 个参数的单一事实源（label、范围、步长、中文说明、分组），`/api/schema` 直接把这张表给前端，前端按 `kind` 分派成滑杆 / 开关 / 分段控件。加参数就改这一处，前后端都会跟上。

### 严格性：宁可报错也不静默

- 未知参数名返回 `400 unknown_parameter`。原先拼错 `strengh=0.7` 会静默用默认值跑出一张"看起来对"的图，这种错误最难查。
- 越界值返回 400，不做 clamp。
- 客户端文件名只用于展示和 `Content-Disposition`，落盘一律 `source<按 magic bytes 嗅探的扩展名>`，路由侧还有 `JOB_ID_RE` + `commonpath` 双重校验，防路径穿越。

### 三个真实踩到的坑

- **7861 不能用**：它落在 `netsh int ipv4 show excludedportrange protocol=tcp` 里 7779–7878 这一段（Hyper-V / WSL 的保留段），绑定直接 `WinError 10013`。默认端口因此是 8791，启动器对 10013 有专门文案。
- **拒绝请求前必须把 body 读掉**：未授权就直接回 401 并关连接，客户端还在写请求体，系统会把它变成一个 RST，客户端看到的是 `ConnectionAbortedError` 而不是 401。浏览器同样表现为网络错误，而不是触发令牌弹窗。`server/api.py:_drain_body()` 在 `_json_error()` 里先把未读的 body 吃掉再回错，`Handler.timeout = 60` 兜住慢客户端。这条在空闲机器上偶发、在负载高时必现，属于典型的「测试跑一次过、跑三次挂一次」。
- **CSS `line-height: 0` 会切碎文字**：`.compare` 用 `line-height: 0` 消掉 inline 图片的基线间隙，结果里面的「原图/修后」标签行盒塌陷成 0 高，字形被切掉一半，放大看是一排彩色碎片。给标签单独设 `line-height: 1.5` 并 `transform: translateZ(0)`（避免继承 `.clipped` 层 `will-change: clip-path` 造成的分数像素挤压）才干净。

### 界面结构

`server/webui/` 是纯静态三件套，没有构建步骤。`index.html` 只有骨架和 id，参数行、诊断表、预设按钮全部由 `app.js` 按 `/api/schema` 生成。

- `app.css` 的暗房配色用 CSS 变量集中在 `:root`，滑杆填充靠 `--fill`、对比滑块靠 `--split`，都不改 DOM 结构。
- 预览图长边 2400px（`--preview` 可调），20 MP 原图直接给浏览器解码要 80 MB，滑块会卡；全分辨率 PNG 走下载按钮。
- 刷新不丢结果是 `localStorage.skin.lastJob` + `?job=<id>`，恢复时会重新拉 `kind=source` 的原始字节，所以恢复后还能直接改参数重跑。
- **任务 id 的服务端契约**：`meta.json` 存的就是完整 snapshot，所以 `server/api.py` 的 `_job_snapshot` / `_meta` / `_result` 都会在内存队列查不到时回落读盘（`_ARTIFACT_FILES` 把 metadata 里的 artifact key 映射回文件名）。重启服务后旧 id 依然 200，这个行为由 `tests/server_test.py` 的「survives a restart」三条断言守着，别为了省事把回落删掉。
- `[hidden] { display: none !important; }` 必须留着：`.busy` / `.gate` / `.dropzone` 都是 `display: grid`，会压过 `hidden` 属性，导致首屏被遮罩糊住。

### 安全边界

回环绑定不需要令牌；非回环必须令牌（不给就随机生成并打印）；`--no-auth` + 非回环直接拒绝启动、退出码 2。令牌接受 `Authorization: Bearer` / `X-Auth-Token` / `?token=` / cookie `skin_token`，用 `hmac.compare_digest` 比对，连续 5 次失败锁 60 秒。`?token=` 登录成功后前端立刻 `history.replaceState` 抹掉地址栏。

### 一键开关

**做了什么**：`服务开关.bat` 双击即切换（没跑就起并开浏览器，在跑就停），背后是 `service.py`——`start / stop / restart / status / open / logs / toggle` 七个动词，只用标准库。PID、端口、令牌、局域网地址记在 `server/data/service.json`，服务端横幅落在 `server/data/service.log`（超过 2 MB 轮转成 `.log.1`）。`tests/service_test.py` 37 项断言全绿，`SERVICE OK 37 checks`。

**为什么这么定**：

- **用 `DETACHED_PROCESS` 起子进程，而不是前台跑**。双击的窗口只是为了报告结果，用户关掉它不该把服务带走；detach 之后同一个 bat 再双击一次就是「停止」，这就是"一键"的全部实现。
- **不 import `server` 包**。`service.py` 只依赖标准库：`import server.api` 会连带加载 cv2 和 torch，那要 2–4 秒，比开关本身该花的时间多两个数量级。token 生成用 `secrets.token_urlsafe(18)`，与 `server/config.py:new_token()` 同一个算法但各自实现，不值得为一行函数付加载代价。
- **自己生成并持久化令牌**。`run_server.py --lan` 会在子进程里生成令牌只打印到横幅，父进程拿不到，`status` 就没法给出可直接点开的地址。所以 `service.py` 在非回环时自己 mint 一个、`--token` 传给子进程、同时写进 state。别人的令牌在 `service.json` 里是明文——这个文件在 `.gitignore` 覆盖的 `server/data/` 下，和上传的图同一级别。
- **`stop` 用 `taskkill /PID /T /F`，不做优雅关闭**。detached 进程没有控制台，收不到 `CTRL_BREAK`；给它开一个 `POST /api/shutdown` 端口等于多一个攻击面。强杀的代价可控：`store.write()` 是 `.part` + `os.replace` 原子写，残留的 `.part` 文件不影响正确性，孤儿目录由 TTL 扫描清掉。
- **只停自己启动的进程**。state 里有 PID 且存活才动手；手动 `run_server.py` 起的实例会被明确告知"找不到它的 PID"，而不是去 `netstat` 猜端口属主然后误杀。pid 会被回收，所以 `pid_alive` 之外还要用 `/api/health` 的 `Server: SkinRetouch/` 头确认对端确实是这个服务。

**没做什么**：

- 没有系统托盘图标。那需要 `pystray` 或 PowerShell 的 WinForms，两者都比"一个 bat"重得多，而零依赖是硬约束。
- 没有开机自启。要的话是往 `shell:startup` 里放一个快捷方式，不需要代码。
- 没有服务化（`sc create`）。Windows 服务要求实现 SCM 协议，和"一个能双击的脚本"是两种东西。
- 没做多实例管理：一个 data 目录只认一个 PID 文件，两个端口要两个 `--data-dir`（`service_test.py` 里的 "a second data dir refuses the busy port" 守着这条）。

**怎么验证**：

```powershell
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\tests\service_test.py   # 期望 SERVICE OK 37 checks
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\service.py status       # 期望 status running / stopped
```

测试在临时数据目录里真起真杀，覆盖：起停闭环、重复 `start` 是空操作且 PID 不变、端口真的释放、进程真的消失、`toggle` 的两个方向、`logs` 能读到横幅、第二个 data 目录拒绝占用中的端口、过期 PID 文件不阻塞启动、`restart` 后磁盘上的任务仍可读。

启动函数的日志编码有个坑值得记：detached 子进程没有控制台，Python 的 stdout 回落到 locale 代码页（这机器上是 cp936），横幅会以 GBK 写进日志，`service.py logs` 用 UTF-8 读就得到一串替换字符。`spawn_server()` 通过 `PYTHONIOENCODING=utf-8` 把子进程的编码钉死，`main()` 再给 stdout 加 `errors="replace"` 兜底——状态查询不该因为一行历史日志就崩。

另一条：客户端中途断开（探活超时、关掉标签页、被 kill 的 curl）在 Windows 上抛的是 `ConnectionAbortedError`（10053），**不是** `ConnectionResetError`。`api.py:_dispatch` 原先只列了后者，于是每次断连都被当成 500 记录一条 traceback，日志里全是噪声。现在两个都列出，`Handler.handle_one_request` 再兜一层。验证方式：连上就关的 socket 打三个请求，日志应当零增长。

---

## 可以继续的方向

1. 暴露色距阈值 `lo/hi` 作为节点参数，救极端光照。
2. 多人场景改成逐人分割 + 逐个滤波。
3. 身体掩膜边界做一次双边重算，压掉浅色衣物的残余误判。
4. `deliver_demo.py` 目前只有一版成品图，可以扩成多套预设的横向对比。

改任何东西之后，跑 `blemish_test.py` 对基线数值。三行 `hf-energy` 有变动就必须解释清楚原因，否则说明算法被动过了。

---

## 文件

```
E:\SkinRetouch\
  HANDOFF.md             本文档
  README.md              面向使用者的方案说明
  __init__.py            节点注册
  skin_retouch_node.py   ComfyUI 封装，处理 RGB / RGBA / 灰度
  skin_core.py           算法核心 + 脸部管线，不依赖 ComfyUI，可单独 import
  skin_body.py           人物分割、身体肤色掩膜（CPU）
  skin_gpu.py            torch 版引导滤波、身体掩膜、LAB 转换
  models/
    face_yolov8m.pt      50 MB 人脸检测器
    person_yolov8m-seg.pt 55 MB 人物分割器
  tests/
    blemish_test.py      注入瑕疵再还原，CPU 脸路径回归基线（带断言）
    node_test.py         节点输入输出契约
    compare_paths.py     CPU / GPU 两条路径的一致性预算
    server_test.py       本地服务 HTTP 端到端自检（54 项断言，随机端口自起自停）
    repaint_test.py      局部重绘自检（15 项断言，引擎不在线时跳过在线部分）
    service_test.py      一键开关起停自检（37 项断言，临时数据目录里真起真杀）
    deliver_demo.py      成品对比图
    probe_body_final.py  20 MP 端到端，出 final_out_/final_mask_/final_side_
    dump_compare.py      1:1 前后对比裁剪
    probe_sweep.py       身体色度窗参数扫描
    probe_*.py           17 个探测脚本，其中 probe_body_final / probe_sweep 在验收链路上，
                         其余 15 个（cluster/sam/gate/skin/regions 系列）是失败路线的取证，
                         留着是为了不重复踩坑，不参与验收
    assets/              测试原图（3 张 896×1152 + 2 张 20.7 MP）
    output/              对照图产物
  run_server.py          图形界面 / 远程接口启动器（argparse + 令牌策略 + 启动横幅）
  服务开关.bat           双击开关（纯 ASCII，避免 cmd 代码页问题）
  service.py             开关本体：start/stop/restart/status/open/logs/toggle，只用标准库
  server/
    __init__.py          把项目根挂进 sys.path，导出 serve()
    config.py            PARAM_SPECS 参数表、PRESETS、Config、令牌与参数校验
    store.py             任务落盘、magic bytes 嗅探扩展名、TTL 清理
    jobs.py              单 worker 串行队列、core_kwargs 参数映射、PNG/JPEG 编码、meta.json
    repaint.py           SDXL 9 通道 inpaint 客户端：掩膜检测、分块规划、alpha 融合、降级
    api.py               http.server 路由、鉴权、CORS、三种上传体、SSE
    webui/               index.html / app.css / app.js / favicon.svg（无构建步骤）
  server/data/           运行时任务目录（.gitignore，可随时删）
```

**自动化验收**：`blemish_test.py` 与 `compare_paths.py` 现在会在数值越界时 `SystemExit(1)` 并打印 `FAIL`，可以直接挂 CI。前者锁 CPU 脸路径（t01/t07/t05 三组 hf-energy 与覆盖率），后者锁两条路径的差距（覆盖率相对差 <3%、成图 p99 <15、均值差 <2）。

`skin_core.py` 的 `skin_retouch` 是总入口，`skin_body.py` 和 `skin_gpu.py` 都可以没有——缺了就自动退回纯脸 CPU 路径（`skin_body` 缺失会发一条 `RuntimeWarning`，不是静默）。关键函数（`_default_face_model`、`face_skin_mask`、`retouch_channel`、`feather_mask`、`_box_mean`、`body_skin_mask`）上方都写了为什么这么写，而不是它做了什么。
