# Skin Retouch (Lite)

只做一件事的 ComfyUI 美肤节点。不重绘、不改五官、不动身份特征，只对**皮肤区域**做保边平滑与色度均匀。（网页端另有一个默认关闭的局部重绘可选项，只在小掩膜上清理定点痘印，见[局部重绘](#局部重绘可选默认关)。）

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
    server_test.py       本地服务的 HTTP 端到端自检（58 项断言）
    repaint_test.py      局部重绘的自检（15 项断言，引擎不在线时自动跳过在线部分）
    service_test.py      一键开关的起停端到端自检（37 项断言）
    deliver_demo.py      成品对比图
    probe_body_final.py  20 MP 端到端与计时
    dump_compare.py      1:1 前后对比裁剪
    probe_sweep.py       身体色度窗参数扫描
    assets/              测试原图
    output/              对照图产物
  run_server.py          图形界面 / 远程接口启动器
  服务开关.bat           双击开关：跑着就停，没跑就起
  service.py             开关本体：start / stop / status / logs / toggle
  server/
    __init__.py          把项目根挂进 sys.path，导出 serve()
    config.py            参数表、预设、Config（13 个参数的单一事实源）
    repaint.py           SDXL 9 通道 inpaint 的客户端：掩膜检测、分块、融合
    engine.py            重绘引擎的生死：探测、复用、按需拉起私有 ComfyUI
    store.py             任务落盘、magic bytes 嗅探、TTL 清理
    jobs.py              单 worker 串行队列、结果编码、meta.json
    api.py               http.server 路由、令牌鉴权、三种上传体
    webui/               index.html / app.css / app.js / favicon.svg
```

`blemish_test.py` 与 `compare_paths.py` 数值越界时会打印 `FAIL` 并以非零码退出，可以直接挂 CI。

## 安装

ComfyUI 用目录链接把它挂进节点列表，代码只维护一份：

```powershell
New-Item -ItemType Junction -Path E:\comfyui\ComfyUI\custom_nodes\ComfyUI-SkinRetouch -Target E:\SkinRetouch
```

不想用链接就把整个目录复制进去，效果一样。

## 图形界面与远程接口

只想修一张照片，不想开 ComfyUI 时用它。一个不依赖任何框架的本地服务：静态网页是界面，同一套 HTTP 接口给脚本、手机、另一台机器用。**没有新增依赖**——`http.server` 加浏览器原生能力，`cv2` / `torch` 之外不需要装任何东西。

```powershell
# 本机：默认 127.0.0.1:8791，回环地址不需要令牌，--open 自动开浏览器
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\run_server.py --open

# 局域网 / 远程：绑定 0.0.0.0，自动生成访问令牌并打印带令牌的地址
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\run_server.py --host 0.0.0.0
```

启动横幅会打印三样东西：本机控制台地址（带 `?token=`）、局域网地址、`/api/jobs` 的调用示例。打开带令牌的链接后令牌被写进 HttpOnly cookie，地址栏里的 `?token=` 立刻被抹掉，不会留在历史记录或截图里。

端口撞车时换一个即可：`--port 8802`。Windows 上 `WinError 10013` 表示该端口被系统保留（Hyper-V / WSL / Docker 会静默占段），不是权限问题，启动器会直接说明。

重绘引擎相关（见[局部重绘](#局部重绘可选默认关)）：`--repaint-url`（默认 `http://127.0.0.1:8188`）、`--repaint-autostart` / `--no-repaint-autostart`（默认开）、`--comfyui-dir`、`--comfyui-python`。启动横幅会打印引擎地址和它这一轮的状态（`standby` 表示已待命，`idle:` 后面跟原因）。

### 一键开关

双击仓库根目录的 `服务开关.bat`：没在跑就启动并打开浏览器，在跑就停掉。同一个脚本连着双击两次等于重启，不需要记命令，也不需要留着那个黑窗口——服务是脱离控制台运行的，窗口关掉服务照跑。

命令行等价物（同一个 `service.py`）：

```powershell
python service.py            # 开关：跑着就停，没跑就起
python service.py start      # 起，等它真的应答了才返回
python service.py stop       # 停，等端口真的释放了才返回
python service.py status     # 状态、地址、健康、PID、运行时长
python service.py logs       # 服务端日志尾部
python service.py open       # 按当前令牌打开控制台
python service.py toggle --lan   # 手机要连时用这个，自动生成令牌并打印局域网地址
```

进程号、端口、令牌记在 `server/data/service.json`，横幅落在 `server/data/service.log`（超过 2 MB 自动轮转一份 `.1`）。所以第二次双击能找到**自己启动的那个进程**去停，而不是去 `netstat` 里猜端口属主。这两个文件都在 `.gitignore` 覆盖的 `server/data/` 里。

停服务会中断正在跑的图——排队的任务直接作废，正在算的那张会被杀掉（`stop` 前会倒计时 3 秒，`-y` 跳过）。已经算完的任务不受影响：`meta.json` 在磁盘上，重启后旧 id 照样能取回结果。

开关只管被它自己启动的进程。手动 `run_server.py` 起的实例，`service.py stop` 会说明找不到它的 PID 并让你去原控制台停——它不去杀一个自己没启动的进程。

### 安全边界

三档，刻意做成不可混淆：

| 场景 | 令牌 | 说明 |
|---|---|---|
| 默认 `127.0.0.1` | 不需要 | 只有本机进程能连，浏览器同源策略也挡住了其它网页 |
| 绑定非回环地址 | **必须** | 没给 `--token` 就随机生成一个并打印，忘了就重启再看 |
| `--no-auth` + 非回环 | 拒绝启动 | 退出码 2，绝不把一块能跑模型的 GPU 裸奔到网络上 |

令牌走 `Authorization: Bearer`、`X-Auth-Token`、`?token=` 或 cookie 四种方式任选其一。连续 5 次错误令牌会锁 60 秒。跨站 POST 默认只认同源，需要别的来源时显式给 `--allow-origin https://example.com`。

### 局部重绘（可选，默认关）

磨皮之后可以再跑一过扩散重绘，只重画检测到的痘印/红斑，用于清理纯滤波压不下去的定点瑕疵。**单个任务默认不跑**，因为代价和前提都很硬：

- 需要一个跑着的 ComfyUI 实例，`--repaint-url` 指定（默认 `$SKIN_REPAINT_URL` 或 `http://127.0.0.1:8188`，即 ComfyUI 的默认端口）。拿不到就跳过，任务照常出图，界面与 meta 里写明跳过原因，不会失败。
- 本机唯一能离线跑的引擎是 SDXL 9 通道 inpaint（`utopianPony_v1Inpainting`）。LaMa / MI-GAN / SD1.5-inpaint 这些真正的小模型本机没有、也装不上，理由与实测见 `REPAINT_EXPLORE.md` 第 1 节。
- 一块 896×896 约 12–24 秒（首块含 6.9 GB checkpoint 加载约 33 秒）。896×1152 的竖图通常命中 2 块，是磨皮本身（0.8 秒）的数十倍。

**引擎待命。** 服务启动时就把重绘引擎拉到可用的状态，不用你先去想 ComfyUI 开没开：

- 8188 上已经有实例（无论你怎么起的）→ 直接复用，**绝不会接管或关闭它**；页面上该引擎的按钮显示「占用中」而不是「停止」。
- 8188 上什么都没有 → 起一个私有实例，约 27 秒待命，日志写到 `<data-dir>/comfyui.log`，服务退出时一并关闭。
- 想省那 2 GB 显存就加 `--no-repaint-autostart`（或 `--repaint-autostart` 显式开启，默认已开）。关闭自动待命后，页面「重绘」组标题旁的圆点仍是灰的，点旁边的「启动」按钮按需拉起；私有实例的日志与输出目录同样落在 data-dir 里。
- 找不到 ComfyUI 安装目录时用 `--comfyui-dir` 指定（含 `main.py` 的那层），解释器用 `--comfyui-python`（默认找隔壁的 `python_embeded`）。找不到时按钮显示「不可用」，提示里写原因。

实测（`tests/probe_web_repaint.py`，注入 26 个痘点 + 10 片红斑的基准图，掩膜命中 2 块）：

| | 瑕疵区残差 | 瑕疵区高频能量 |
|---|---|---|
| 磨皮后 | 4.80 | 6.46 |
| 再加一过重绘 | **4.10** | **5.46** |

掩膜外像素逐字节不变（`tests/repaint_test.py` 断言 `changed_outside == 0`），发丝与五官不受影响。**已知短板在自动检测不在引擎**：自动掩膜只覆盖注入痘点面积的 45%，剩余靠磨皮兜着；换成真值掩膜，同一个引擎能把瑕疵区残差从磨皮后的 6.20 压到 4.53、高频从 6.57 压到 4.44（`tests/probe_hit_check.py`）。

三个参数：`repaint`（开关）、`repaint_denoise`（0.60，重绘幅度）、`repaint_steps`（8，采样步数——8 步与 20 步在本场景没有可测差异）。

引擎的启停走 `GET /api/repaint/engine`（读状态）与 `POST /api/repaint/engine`（body `{"action": "start"|"stop"}`），状态字段为 `state`（`idle`/`starting`/`ready`/`failed`）、`available`、`managed`（是否由本服务拉起）、`spawnable`、`detail`。`start` 立即返回 `starting`，加载完成前请按 1–2 秒的间隔轮询。

### 界面能做什么

- **三档预设 + 13 个参数**，滑杆实时显示数值，参数说明直接写在下面，不用回 README 找含义。
- **对比滑块**：原图与修后叠在一起，拖动分割线看差异，键盘方向键可微调；另有单图视图、掩膜视图、诊断视图。
- **掩膜视图**回答"它到底动了哪里"：白色是会被美肤的像素，五官保留为黑。
- **诊断视图**给出分辨率、三段耗时条形图，以及每张脸/身体的位置、覆盖比例、羽化半径、分割条带数。
- 大图自动生成 2400px 长边的预览图，浏览器不卡；**下载按钮给全分辨率 PNG**。
- 刷新页面不丢结果（`?job=<id>` 也能直接打开某次结果），排队中的任务可以撤退单。
- 快捷键：`Ctrl/⌘ + ↵` 处理、`1/2/3/4` 切视图、`z` 切适应/1:1。

### 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/health` | 存活、CUDA、模型是否已加载、队列深度 |
| GET | `/api/schema` | 参数表、预设、默认值、上传上限 |
| POST | `/api/login` | 用令牌换 cookie |
| POST | `/api/jobs` | 提交一张图，`202` + `Location` 头 |
| GET | `/api/jobs` | 任务列表（含排队中） |
| GET | `/api/jobs/<id>` | 状态快照：阶段、耗时、info、错误 |
| GET | `/api/jobs/<id>/events` | SSE 流，脚本或第三方前端可用 |
| GET | `/api/jobs/<id>/result` | `kind=out\|mask\|source\|preview_out\|preview_mask\|preview_source`，`&download=1` 带文件名 |
| GET | `/api/jobs/<id>/meta` | 落盘的 `meta.json` |
| DELETE | `/api/jobs/<id>` | 撤退还在排队的任务；已在跑的不假装能中断 |

提交支持三种请求体，按调用者方便来选：

```bash
# 1. 原始字节 + query 参数（浏览器与大多数脚本）
curl -X POST "http://127.0.0.1:8791/api/jobs?strength=0.75&include_body=1&name=portrait.jpg" \
     -H "Content-Type: image/jpeg" --data-binary @portrait.jpg

# 2. multipart/form-data（想混传参数 JSON）
curl -X POST http://127.0.0.1:8791/api/jobs \
     -F "image=@portrait.jpg" -F 'params={"strength":0.9,"blemish":0.85}'

# 3. base64 JSON
curl -X POST http://127.0.0.1:8791/api/jobs \
     -H "Content-Type: application/json" \
     -d "{\"image\":\"$(base64 -w0 portrait.jpg)\",\"name\":\"p.jpg\"}"

# 远程带令牌轮询
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8791/api/jobs/<id>
```

参数名与节点完全一致（`face_confidence`、`include_body`、`body_strength=0` 表示跟随 `strength`），参数拼错会返回 `400 unknown_parameter` 而不是悄悄用默认值跑出一个看起来对的结果。越界值同样 400，不做静默夹取。

### 数据与清理

上传的原图、成图、掩膜、预览图、`meta.json` 全部落在 `server/data/jobs/<id>/`，客户端文件名只作展示、从不用来拼路径。默认 1 小时后清理（`--ttl` 改），关掉服务也可以直接删 `server/data/`。这个目录在 `.gitignore` 里。

任务 ID 是可寻址的资源：`meta.json` 里存着完整快照，所以服务重启后内存队列虽然空了，拿着旧 id 仍然能取回状态、诊断数据和全部图片（页面刷新也能靠 `?job=<id>` 恢复上一次的结果）。

### 自检

```powershell
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\tests\server_test.py    # 期望 SERVER OK 54 checks
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\tests\service_test.py   # 期望 SERVICE OK 37 checks
```

第一个会在随机端口拉起一个实例，跑完 54 项断言后关掉：鉴权、坏参数、路径穿越、三种上传体、结果图尺寸、预览缩放比例、SSE、下载头、meta 落盘，以及重绘开关的三种情形（关、引擎不在线时降级、引擎在线时真的跑一过）。

第二个针对一键开关本身，在临时数据目录里真的启动和杀死进程，37 项断言覆盖：起、停、状态、日志、重复启动是空操作、端口真的释放、PID 真的没了、`toggle` 两个方向、过期 PID 文件不阻塞启动，以及重启后磁盘上的任务仍可读。

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
