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

**1. 不用扩散模型重绘。**
美肤是信号处理问题，不是重绘问题。任何扩散或 inpainting 模型都会重画皮肤，重画的代价是人脸身份漂移——眼神、牙形、痣的位置会变。生成模型出的图本来就干净，再拿它重绘一遍纯属毁图。用户明确只要美肤这一项。

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
    deliver_demo.py      成品对比图
    probe_body_final.py  20 MP 端到端，出 final_out_/final_mask_/final_side_
    dump_compare.py      1:1 前后对比裁剪
    probe_sweep.py       身体色度窗参数扫描
    probe_*.py           17 个探测脚本，其中 probe_body_final / probe_sweep 在验收链路上，
                         其余 15 个（cluster/sam/gate/skin/regions 系列）是失败路线的取证，
                         留着是为了不重复踩坑，不参与验收
    assets/              测试原图（3 张 896×1152 + 2 张 20.7 MP）
    output/              对照图产物
```

**自动化验收**：`blemish_test.py` 与 `compare_paths.py` 现在会在数值越界时 `SystemExit(1)` 并打印 `FAIL`，可以直接挂 CI。前者锁 CPU 脸路径（t01/t07/t05 三组 hf-energy 与覆盖率），后者锁两条路径的差距（覆盖率相对差 <3%、成图 p99 <15、均值差 <2）。

`skin_core.py` 的 `skin_retouch` 是总入口，`skin_body.py` 和 `skin_gpu.py` 都可以没有——缺了就自动退回纯脸 CPU 路径（`skin_body` 缺失会发一条 `RuntimeWarning`，不是静默）。关键函数（`_default_face_model`、`face_skin_mask`、`retouch_channel`、`feather_mask`、`_box_mean`、`body_skin_mask`）上方都写了为什么这么写，而不是它做了什么。
