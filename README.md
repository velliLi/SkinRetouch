# Skin Retouch (Lite)

只做一件事的 ComfyUI 美肤节点。不重绘、不改五官、不动身份特征，只对**皮肤区域**做保边平滑与色度均匀。

处理范围是**主要人物的脸 + 露出来的身体皮肤**：脖子、肩、手臂、手、胸口。目标画幅是 20 MP 级的全身或半身照（5568×3712 / 3712×5568）。

## 为什么不是扩散模型

美肤在生成式管线里被当成"重绘任务"来做（inpaint 脸部重画），代价是：

- 身份漂移。重画一次，脸就不是原来那张脸了。
- 每张数秒。6GB 显存还要和主模型抢。
- 无法局部控制。用户要"皮肤好一点"，模型给的是"换一张脸"。

美肤本质是一个**信号处理**问题：皮肤缺陷（痘、斑、毛孔、泛红）都落在特定频率带和色度通道上，而五官结构（睫毛、唇线、鼻翼、发丝）是高频强边缘。把两者在频域上分开，就能只动前者。这也是商业美颜 SDK 的实际做法。

## 模型与依赖

| 项 | 大小 | 来源 | 说明 |
|---|---|---|---|
| `face_yolov8m.pt` | 50 MB | 本机 `models/ultralytics/bbox/` 已有 | 定位人脸框 |
| `person_yolov8m-seg.pt` | 55 MB | 本机 `models/ultralytics/segm/` 已有 | 定位主人物轮廓，只在 `include_body=True` 时加载 |
| `cv2.ximgproc.guidedFilter` / torch CUDA | 无 | opencv-contrib / torch 2.5.1+cu121，已装 | 保边平滑核心算子 |
| numpy / opencv / torch | 无 | ComfyUI 环境自带 | 无新增依赖 |

**不新增任何依赖，不需要联网。** 显卡可选：`device="auto"` 时用 CUDA，不可用就退回 CPU（慢 2.7 倍）。

两个模型的作用都被压到最小的必要条件：把人脸和人物各框出来，限定搜索范围。皮肤与非皮肤的像素级判定交给自适应肤色模型与方向性纹理判据，那是确定性的数学，可复现、可解释、无幻觉。

## 处理链路

对每张脸依次执行：

1. **人脸定位** — YOLOv8-face 出框，取置信度最高的 N 张。每张脸随后只在自己的 ROI（框外扩 30%）内计算，与画幅大小解耦。
2. **皮肤掩膜** — 三层相乘：
   - 脸型椭圆先验（裁掉框内背景、发际线以上）
   - 自适应 YCrCb 色距，参考色取自左右脸颊 + 鼻梁 + 下巴四块的中位数
   - 结构抑制：远低于肤色亮度的像素（眼、眉、睫毛、鼻孔影）与高饱和像素（唇红）降权
3. **级联尺度空间平滑**（在 CIELab 的 L 通道，导向滤波保边）：
   - `fine = G(I, r≈脸宽×1%)` — 去掉毛孔以下的传感器噪声
   - `mid = G(fine, r≈脸宽×3~7%)` — 去掉 3–15px 的痘印，得到干净基底
   - 中频带（8–30px，正是痘与斑的尺度）丢弃
   - 高频带按**幅度自适应**衰减：`exp(-(hf/τ)²)` 在平坦噪声处≈1、在真实结构处≈0
4. **色度均匀** — a/b 通道向大半径局部均值收敛，压掉泛红与色斑，不改变肤色均值。
5. **羽化与亮度守恒** — 掩膜边缘高斯羽化（半径随脸大小缩放），合成后把掩膜内的平均亮度对齐回原图，避免整体提亮或发灰。

关键点：**亮度守恒**和**幅度自适应**这两条，是它不像"塑料脸"的原因。

身体皮肤走第二遍，步骤不同：

1. **主人物分割** — `person_yolov8m-seg` 在 640 上出轮廓，取面积最大者。它只负责"哪些像素属于这个人"。
2. **身体肤色掩膜** — 四层相乘：
   - 参考色**锚定人脸**（人的身体和脸是同一个人的肤色，这是唯一可靠的先验）
   - 色度椭圆门，半径远小于脸部（`Cr±9 / Cb±7`）
   - 高光补偿：Cr 放开 3.2 倍、Cb 只放开 1.6 倍——过曝皮肤的 Cb 只偏 +5，浅色衣服偏 +12 以上
   - 方向性纹理抑制：织物条纹让梯度集中在单一方向（`aniso` 显著为正），皮肤接近各向同性
3. **全分辨率滤波** — 尺度由最大人脸决定，一次上卡算完整幅人物。
4. **脸部挖空** — 身体掩膜在脸部被砍掉，避免同一块皮肤被磨两遍。

已知不可分的情况：浅粉/肉色紧身衣的某些区域与皮肤色度完全一致。那块布料本身平滑，美肤对它的实际改动接近零，1:1 对比看不出差异。

## 节点参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `strength` | 0.75 | 总强度 |
| `texture_keep` | 0.35 | 保留的毛孔级纹理。0 = 塑料，0.6 = 几乎没动 |
| `blemish` | 0.7 | 痘印去除半径，越大清得越大片 |
| `tone_even` | 0.5 | 泛红/色斑均匀化 |
| `feather` | 0.5 | 掩膜边缘软度，相对脸大小 |
| `max_faces` | 4 | 处理的最大人脸数 |
| `face_confidence` | 0.35 | 检测阈值 |
| `include_body` | True | 连身体皮肤一起处理。关掉就只剩脸，也就不用显卡 |
| `body_strength` | 0.0 | 身体强度。0 = 跟随 `strength`；全身照建议比脸低 0.1~0.2 |
| `device` | `auto` | `auto` / `cuda` / `cpu` |

`include_body` 有两个默认值：节点上 `True`，而 `skin_core.skin_retouch()` 的函数默认是 `False`。这是刻意的——库级默认保持向后兼容，老代码升级后逐像素不变；节点面向使用，默认把身体带上。直接 import 调用想处理身体，必须显式传 `include_body=True`。

`device` 也有一对：节点 `"auto"`，函数 `None`（等价 `"cpu"`）。所以 `tests/blemish_test.py` 那三组基线是 **CPU 路径** 的产物，GPU 路径由 `tests/compare_paths.py` 单独守着。

输出 `image` 与 `skin_mask`（掩膜可接 inpaint 或用于排查）。

预设：

- 轻度（保留质感）：`strength 0.5, texture_keep 0.55, blemish 0.5`
- 标准（默认）：`strength 0.75, texture_keep 0.35, blemish 0.7`
- 强（商业修图）：`strength 0.9, texture_keep 0.18, blemish 0.85, tone_even 0.7`

## 实测

`include_body=False`，纯 CPU，含人脸检测的稳态耗时：

| 图幅 | 人脸 | 耗时 |
|---|---|---|
| 448×576 | 313 px | 274 ms |
| 896×1152 | 778 px | 457 ms |
| 1344×1728 | 1167 px | 743 ms |
| 2784×1856 | 1743 px | 1.44 s |
| 5568×3712（20 MP 半身） | 1538 px | 1.47 s |
| 5568×3712（20 MP 满幅特写） | 3487 px | 5.0 s |

耗时跟着**人脸像素尺寸**走，不跟画幅走：20 MP 半身照和 5 MP 半身照都在 1.4 s 上下，因为滤波只在人脸 ROI（框外扩 30%）里做，画幅多出来的背景根本不参与计算。

身体路径无法缩小计算域，必须在整幅人物上算。两张 20.7 MP 示例图热态实测：

| 阶段 | DSC_7300（竖） | DSC_4081（横） |
|---|---|---|
| 人脸检测 + 逐脸处理 | 238 ms | 205 ms |
| 人物分割 | 694 ms | 940 ms |
| 身体掩膜 | 742 ms | 797 ms |
| 身体滤波 | 609 ms | 579 ms |
| 羽化、合成、均值对齐 | 1371 ms | 1767 ms |
| **合计（GPU）** | **3.65 s** | **4.29 s** |

两张同为全身照却差了 0.6 秒：DSC_4081 的人物掩膜更碎（背景里还有八个人），分割和羽化都更贵。

首次调用多 **20 s** 左右（两个模型从磁盘加载 7 s + 一次性 CUDA 上下文与算子初始化），之后常驻。接进 ComfyUI 时第一次执行会明显卡住，不是死循环。

### 关于 GPU

检测不用 GPU：20 MP 的瓶颈是 letterbox 预处理的 `cv2.resize`，实测 CPU 186 ms / CUDA 190 ms，一样。

滤波必须用 GPU：`cv2.ximgproc.guidedFilter` 没有 CUDA 后端，本机 OpenCV 也是纯 CPU 构建。20 MP、r=66 的导向滤波 cv2 要 410 ms，`skin_gpu.py` 里自己写的 torch 版 61 ms。全身美肤的总差距是 9.8 s → 3.65 s。

CUDA 不可用时自动降级到 CPU，结果**接近但不完全相同**：矩形核形态学与 YUV 转换的浮点舍入让掩膜覆盖率差约 1.5%（GPU 0.1322 / CPU 0.1342）。视觉上无差异，逐像素对比会有差别。

原图本身已被生成模型修过，看不出差异，所以验证方式是**注入已知瑕疵再还原**：在皮肤掩膜内注入 26 个痘点、10 片低频红斑、σ=3 高斯颗粒，然后对比。

皮肤区内高频能量（残留瑕疵与结构的混合度量）：

| 图 | 干净原图 | 高斯模糊基线 | 本管线 |
|---|---|---|---|
| t01 | 5.9 | 1.3 | 7.5 |
| t07 | 6.1 | 2.0 | 9.4 |
| t05 | 7.4 | 2.0 | 8.3 |

高斯模糊把高频压到原图的 20–30%，那是结构被一起抹掉的证据（牙齿、唇纹、鼻翼全糊）。本管线保留结构，高于原图的差额来自注入颗粒未被完全消除。

视觉对照见 `tests/output/*_compare.png`：第 1 栏注入瑕疵、第 2 栏高斯模糊、第 3 栏本管线、第 4 栏参考。第 1、3 栏对照可见红斑与痘点消失而鼻翼/唇纹/牙齿边缘完好。

## 已知限制

- **身体只处理主人物**。多人合影里第二个人之后的身体不碰（脸仍然处理，`max_faces` 管）。
- 浅色紧身衣会和皮肤色度混叠，见上面「已知不可分」。
- 侧脸、遮挡、口罩、小脸（<150px）时掩膜质量下降。全身照里的小脸覆盖率会明显偏低。
- 极深肤色或强单色光（如纯红氛围灯）下，自适应肤色阈值需要调低 `face_confidence` 之外的手动介入，目前没有暴露色距阈值参数。
- 不做肤色美白/改色调，只做均匀化。要提亮请接色彩节点。

## 环境备注

本机 `mediapipe 0.10.14` 与 `protobuf 5.29.5` 不兼容，`FaceDetection` / `FaceMesh` 初始化即崩溃：

```
[libprotobuf ERROR text_format.cc:335] Error parsing text-format
mediapipe.CalculatorGraphConfig: 15:22: Expected identifier
RuntimeError: Failed to parse
```

所以没有走 landmark 路线。降级 protobuf 会影响 `onnx 1.20`，升级 mediapipe 需要动 `python_embeded`，都不划算。当前方案绕开了这个坑。

## 文件

```
E:\SkinRetouch\
  __init__.py            节点注册
  skin_retouch_node.py   ComfyUI 封装
  skin_core.py           算法核心 + 脸部管线，无 ComfyUI 依赖，可单独 import
  skin_body.py           人物分割、身体肤色掩膜
  skin_gpu.py            torch 版引导滤波与身体掩膜，没有它自动退回 CPU
  models/
    face_yolov8m.pt      50 MB 人脸检测器
    person_yolov8m-seg.pt 55 MB 人物分割器
  tests/
    blemish_test.py      注入已知瑕疵再还原的对照实验（带基线断言）
    node_test.py         节点输入输出契约
    compare_paths.py     CPU / GPU 两条路径的一致性预算
    deliver_demo.py      成品对比图
    probe_body_final.py  20 MP 端到端与计时
    dump_compare.py      1:1 前后对比裁剪
    probe_sweep.py       身体色度窗参数扫描
    assets/              测试原图
    output/              对照图产物
```

`blemish_test.py` 与 `compare_paths.py` 数值越界时会打印 `FAIL` 并以非零码退出，可以直接挂 CI。

## 安装

ComfyUI 用目录链接把它挂进节点列表，代码只维护一份：

```powershell
New-Item -ItemType Junction -Path E:\comfyui\ComfyUI\custom_nodes\ComfyUI-SkinRetouch -Target E:\SkinRetouch
```

不想用链接就把整个目录复制进去，效果一样。

## 单独运行

不依赖 ComfyUI，只要 OpenCV + ultralytics：

```python
import sys; sys.path.insert(0, r"E:\SkinRetouch")
import cv2, numpy as np, skin_core
bgr = cv2.imdecode(np.fromfile("in.png", np.uint8), cv2.IMREAD_COLOR)
out, mask, info = skin_core.skin_retouch(bgr, strength=0.75, texture_keep=0.35,
                                         include_body=True, device="auto")
cv2.imencode(".png", out)[1].tofile("out.png")
```
