# SkinRetouch 优化审计报告

审计对象：`E:\SkinRetouch`（ComfyUI 美肤节点 + 可独立 import 的库）
审计日期：2026-02-11
审计方式：静态通读 + 四轮仪器化实测（诊断脚本跑完已删除，未留任何产物）
样本：`tests/assets/DSC_7300.JPG`（3712×5568，20.7 MP，竖幅全身），人物掩膜覆盖率 0.127

---

## 0. 结论

有优化空间。但**优先级最高的不是性能，而是 CPU 与 GPU 两条路径已经语义分叉**：同一个 `skin_retouch` 调用，`device="auto"` 和 `device="cpu"` 的输出逐像素最大差 76 灰阶，掩膜有 1.6% 的像素差值超过 32 灰阶。而三条验收基线（t01/t07/t05）由 CPU 路径产生，意味着 GPU 路径（真正在用的那条）从未被基线约束过。

性能侧的现状是 3.8 秒/帧（20.7 MP，GPU 热态）。文档记载的 3.65 秒仍然可信，但分阶段明细整体偏低 18–34%，不能再作为预算依据。

工作量分布：P0 三项都是几行改动，不碰算法；P1 两项各约十行；P2 一项是替换一个函数体。真正花钱的是 P3 的回归基线建设。

---

## 1. 实测现状基线

环境：`E:\comfyui\python_embeded\python.exe`（Py 3.11.9）、RTX 3060 Laptop 6 GB、`torch 2.5.1+cu121`、`cv2 4.10.0`（纯 CPU 构建）、`ultralytics 8.3.40`。热态（warm≥1，CUDA 同步后计时）。

### 1.1 端到端

| 场景 | 实测 | HANDOFF 记载 | 偏差 |
|---|---|---|---|
| `include_body=True`, `device="auto"` | 3812 ms | 3650 ms | +4.4% |
| `include_body=True`, `device="cpu"` | 10029 ms | 9800 ms | +2.3% |
| `include_body=False`, `device="auto"` | 1135 ms | — | — |

### 1.2 分阶段（`device="auto"`，取自 `info` 返回）

| 阶段 | 实测 | HANDOFF 记载 | 偏差 |
|---|---|---|---|
| 人脸阶段 `faces` | 205.5 ms | 238 ms | −13.7% |
| 人物分割 `person` | 818.5 ms | 694 ms | +17.9% |
| 身体掩膜 `mask` | 996.2 ms | 742 ms | +34.3% |
| 身体滤波 `body` | 770.1 ms | 609 ms | +26.5% |

同一阶段在 CPU 下的数字：`person` 820.1 / `mask` 2835.0 / `body` 4805.8 / `faces` 197.9 ms。

身体滤波的裸测值：GPU 整幅 `retouch_image` 399.7 ms，CPU 条带 `_retouch_body_strips` 4778.6 ms（9 个条带）——**11.8 倍差距**。

### 1.3 CPU 身体掩膜 2835 ms 的构成

| 子阶段 | 耗时 |
|---|---|
| `chroma_gate` | 888.1 ms |
| `texture_suppression` | 464.8 ms |
| `close_31`（31×31 闭运算） | 247.7 ms |
| `bgr2hsv` float 转换 | 99.1 ms |
| `bgr2ycrcb` float 转换 | 89.8 ms |
| `open_11` | 34.0 ms |
| `erode_9` | 13.3 ms |
| `fill_small_holes` | 变动成本，见 P1-1 |

### 1.4 CPU / GPU 一致性

| 指标 | 数值 |
|---|---|
| 掩膜覆盖率 CPU / GPU | 0.1302 / 0.1275（相对差 2.07%，文档称 1.5%） |
| 掩膜平均绝对差 | 1.79 灰阶 |
| 掩膜差 >32 灰阶的像素占比 | 1.6% |
| 成图逐像素最大差 | 76 灰阶 |
| 成图逐像素平均差 | 0.96 灰阶 |

均值小、尖峰大，说明差异集中在边界与色阶过渡带，不是全局偏移。

---

## 2. P0 正确性

### P0-1 色度均匀化半径差一倍

`skin_core.retouch_face` 用 `cv2.blur(a, (r_ch, r_ch))`，核宽 = `r_ch`。
`skin_gpu.retouch_image:304-305` 用 `box_blur(a, r_ch)`，而 `box_blur` 是半径语义，核宽 = `2*r_ch+1`。

实测（r_ch=169）：CPU 支撑半径 84，GPU 支撑半径 169。**色度均匀化的空间尺度差整整一倍**，这是 1.4 节那 76 灰阶尖峰的主因之一。

基线由 CPU 产出（见 2.4 节），所以要改的是 GPU 侧：

```python
# skin_gpu.py:304-305
box_blur(a, max(1, r_ch // 2))
# 同批 301 行的 r_ch 计算保持不变，只改传参语义
```

### P0-2 tau 两套算法

`skin_core.retouch_channel` 内部逐 ROI / 逐条带计算 `tau = max(2.0, 2.5*_mad_sigma(highband[active]))`。
`skin_gpu.retouch_image:292-296` 用全幅单一标量，且高频基准拿 `_box_mean` 代替 guided filter：

```python
fine_for_tau = _box_mean(L, r_fine)
high_for_tau = L - fine_for_tau
tau = mad_sigma(high_for_tau, active)
```

后果：同一次调用里，脸与身体、GPU 与 CPU 会拿到不同的高频抑制强度，`w_edge = exp(-(highband/tau)**2)` 随之漂移。目前这个差异完全隐式，两条路径谈不上可比。

处理方式二选一：把 tau 提成调用方显式传入的参数（GPU 侧的 `retouch_channel` 已经这么做了，CPU 侧对齐即可），或在 `HANDOFF.md` 里明写“两条路径输出不可互换，不要交叉比较”。

### P0-3 偶数核会放大张量并硬崩

`skin_gpu.py:176-183`：

```python
def _dilate(t, k): return F.max_pool2d(t, k, 1, k // 2)
def _erode(t, k):  return -F.max_pool2d(-t, k, 1, k // 2)
```

偶数 k 时 `out = H + 2*(k//2) - k + 1 = H + 1`。实测 640×960 传 k=10 得 641×961，随后 `soft * pm_bin` 抛：

```
RuntimeError: The size of tensor a (129) must match the size of tensor b (128) at non-singleton dimension 3
```

默认值 `open_px=11 / close_px=31 / edge_shrink=9` 全是奇数，所以从未暴露。但 `body_skin_mask_gpu` 的这组参数是公开签名，调用方传偶数即崩。

```python
def _dilate(t, k): k = int(k) | 1; ...
def _erode(t, k):  k = int(k) | 1; ...
```

### P0-4 验收基线绑在非默认路径上

`tests/blemish_test.py:108` 调用 `skin_core.skin_retouch(bad, strength=..., texture_keep=..., blemish=..., tone_even=..., feather=...)`，**不传 `device`**。而 `skin_retouch` 的签名是 `device=None`，函数体里 `if device is None: device = "cpu"`。所以：

- 库级默认是 CPU
- 节点默认是 `"auto"`（`skin_retouch_node.py:54-56`）
- 三行 hf-energy 基线产自 CPU

这个双默认 `HANDOFF.md` 只提了 `include_body` 那一处，`device` 这处没写。GPU 路径因此完全落在基线的约束之外。要么给基线补一组 GPU 数字，要么在文档里把“基线 = CPU 路径”写成显式前提。

---

## 3. P1 稳定性

### P1-1 填洞是 O(K·H·W)

`skin_body.py:158-165`：

```python
for i in range(1, num):
    ...
    out[labels == i] = 255
```

每个候选洞做两次全图布尔运算。实测：无洞时纯付 149 ms（`connectedComponentsWithStats` 本身的开销，2 个标签时直接 return）；人工种 300 个 24×24 的洞后 528 ms。极端画面（大面积花纹衣物、碎发）会让 K 上升，耗时线性恶化到秒级。这是目前代码里唯一一个能在真实输入上从毫秒滑到十秒量级的路径。

改法（实测 528 ms → 192 ms，输出位级一致）：

```python
keep = np.zeros(num, bool)
for i in range(1, num):
    x, y, bw, bh, area = stats[i]
    if area > max_hole_px or x == 0 or y == 0 or x + bw >= w or y + bh >= h:
        continue
    keep[i] = True
ids = np.nonzero(keep)[0]
if ids.size:
    out = mask_u8.copy()
    out[np.isin(labels, ids)] = 255
    return out
return mask_u8
```

那 149 ms 的固定开销还在。要再省，先用 1/4 降采样跑一遍连通域做预检，无候选就跳过全分辨率分析。注意 GPU 路径同样在付这笔钱（`skin_core.py:413` 在 GPU 掩膜之后紧跟着调 CPU 版填洞）。

### P1-2 `skin_body` 导入无保护，与文档矛盾

`skin_core.py:390` 是裸调用：

```python
if include_body:
    skin_body = _optional_module("skin_body")
```

`skin_gpu` 的两处调用（`406`、`444`）都包在 try/except 里，`skin_body` 这处没有。而 `HANDOFF.md:292` 明写：

> `skin_body.py` 和 `skin_gpu.py` 都可以没有——缺了就自动退回纯脸 CPU 路径。

实测行为：把 `skin_body.py` 移走，`include_body=True` 直接抛 ImportError，整帧失败。文档与代码二者必须改一个。代码侧改法：

```python
if include_body:
    try:
        skin_body = _optional_module("skin_body")
    except Exception:
        include_body = False
```

### P1-3 GPU 失效被静默吞掉

两处 `except Exception:` 把 GPU 失败降级成 CPU 回退（`skin_core.py:416` 的 `bmask = None`、`453` 的 `fixed = None`），无任何日志，只能在返回的 `info["backend"]` 里看到 `"cpu"`。代价是 3.8 s 变 10 s，而用户看到的是“节点变慢了”，不是“显卡没被用上”。

至少 `warnings.warn(f"skin_gpu unavailable, falling back to CPU: {e}")` 一次。用 `warnings` 而不是 `print`，避免刷屏。

---

## 4. P2 性能

### P2-1 CPU 路径不建议优化

`mask` 阶段 CPU 2835 ms vs GPU 996 ms。拆解显示大头是 `chroma_gate`（888）、`texture_suppression`（465）、`close_31`（248），而这三者在 `skin_gpu` 里都有等价实现。CPU 路径是回退路径，投入产出比低。

该做的是让回退可见：设备不可用时给明确告警（见 P1-3），而不是让人误以为 10 秒是正常速度。若确实需要可用的纯 CPU 方案，方向是把 `chroma_gate` 与 `texture_suppression` 的数据类型从 float32 降到 uint8/int16 分步运算，而不是改算法。

### P2-2 条带裁剪无收益（已验证否掉）

CPU 身体滤波按 640 行切条，pad 为 `max(192, max(r_mid, r_ch)+16)`。直觉上可以按人物 bbox 横向裁剪，但实测该样本的皮肤横向覆盖 **98.5% 帧宽**（bbox `[0, 468, 3656, 5567]`），裁了几乎不省。这条路对这张图无效。对窄幅人像可能有效，但没有普适收益，不值得为它引入分支。

### P2-3 节点侧转换，改进空间有限

`skin_retouch_node.py:81` 的 `(src*255.0).round().astype(np.uint8)` 实测 160.8 ms；换成 `np.clip(src*255.0+0.5, 0, 255).astype(np.uint8)` 反而 239.8 ms（多的那次 clip 遍历比 round 贵）。两者输出逐像素一致。**保持现状**。

`:95` 的 `mask.astype(np.float32)/255.0` 40.3 ms，换乘 `(1/255)` 39.7 ms，无意义。

---

## 5. P3 工程化

### P3-1 零断言、零 pytest、零 CI

`tests/` 下 19 个 `probe_*.py` 加上 `blemish_test.py`、`node_test.py`、`deliver_demo.py`、`dump_compare.py`、`bench_gpu.py`，全部是打印式脚本。身体路径与 GPU 路径没有任何回归基线；三条 hf-energy 靠人眼比对三位小数。

投入产出比最高的一件事：把基线落成 JSON，加阈值断言。

```python
BASE = {"t01": (5.9, 1.3, 7.5, 0.320), "t07": (6.1, 2.0, 9.4, 0.264), "t05": (7.4, 2.0, 8.3, 0.011)}
got = (hf_clean, hf_blur, hf_ours, coverage)
for g, b in zip(got, BASE[name]):
    assert abs(g - b) <= tol, f"{name}: {got} != {BASE[name]}"
```

再补一组 GPU 与 CPU 的一致性断言（掩膜覆盖率相对差 <3%，成图 p99 逐像素差 <12），P0-1 这类分叉才有办法被自动抓住。

### P3-2 `bench_gpu.py` 不能当证据

`bench_gpu.py` 里的 `box_sum` 用的是二维二次 cumsum——正是被 `_box_mean` 淘汰掉的那个实现。`HANDOFF.md` 记录了它的问题：20 MP 上中间值累积到 `255*W*H ≈ 5.3e9`，float32 只有 7 位有效数字，产生 Z 形等值线。这个脚本给出的 GPU 滤波耗时既不代表生产实现，还可能把早已修掉的精度 bug 当作“基准”复现出来。引用它之前先把它对齐到分离式一维 cumsum。

### P3-3 文档与实测的偏差

| 项目 | 文档 | 实际 |
|---|---|---|
| `mask` 阶段 | 742 ms | 996 ms |
| `body` 阶段 | 609 ms | 770 ms |
| `person` 阶段 | 694 ms | 818 ms |
| CPU/GPU 覆盖率差 | 1.5% | 2.07% |
| `skin_body` 可缺省 | “缺了自动退回纯脸” | 抛 ImportError |
| `device` 默认值 | 未记载 | 库 `"cpu"` / 节点 `"auto"` |
| `detect_primary_person` 选人策略 | “面积最大且置信度最高” | 仅面积最大（`skin_body.py:133`） |
| 测试脚本数量 | 列 7 个 | 实有 19 个 probe |

`skin_body.py:133` 的 `idx = int(np.argmax(areas))` 只看面积，`scores` 算出来只用于回报。要么改文档，要么真的加置信度权重。

---

## 6. P4 清洁度

| 位置 | 问题 |
|---|---|
| `skin_core.py:507-509` | `sub` 是视图，末行 `result[ry1:ry2, rx1:rx2] = sub` 是自拷贝，删掉 |
| `skin_core.py:426, 469` | `body_scale_px(bgr, max_face_px)` 在同一个 body 分支里调了两次，第二次纯粹为了填 `info` |
| `skin_gpu.py:291` | `probe = t[:, 0:1] * 0.0` 后接 `del probe`，死代码 |
| `skin_core.py` `detect_faces` | `restore()` 返回值被丢弃 |
| `skin_core.py:76` vs `skin_body.py:60-65` | `_warm_cuda()` 与 `load_person_segmenter` 内联的 torch 预热是两套 |
| `skin_core.py` `_keep_cuda_visible` vs `skin_body.py:116-126` | CUDA_VISIBLE_DEVICES 保护两套实现，语义一致但维护点翻倍 |
| `skin_core.load_face_detector` | 无模型存在性检查，直接 `YOLO(path)`；`load_person_segmenter` 有 `FileNotFoundError` 友好提示 |
| `skin_core._reference_skin` | 四点 patch 像素池化后取中位数，文档写成“四点中位数”，语义不同 |

前四条是删除级改动，后四条是抽公共函数级。

---

## 7. 动手顺序

| 序 | 事项 | 规模 | 理由 |
|---|---|---|---|
| 1 | P0-3 偶数核断言 | 2 行 | 封死唯一一处硬崩溃 |
| 2 | P0-1 色度半径对齐 GPU | 1 行 | 消除最大的输出分叉，风险低 |
| 3 | P1-1 填洞改 `np.isin` | ~10 行 | 位级一致，最坏情况从秒级回毫秒 |
| 4 | P1-2 `skin_body` 导入加保护 | 4 行 | 让文档承诺成立 |
| 5 | P1-3 静默 except 加告警 | 2 行 | 让回退可见 |
| 6 | P3-1 基线断言化 | 中 | 前五项的安全网 |

前五项都不碰算法，改完跑一遍 `blemish_test.py` 应保持三条基线逐位不变（因为基线走 CPU，只受 P1-1 影响，而 P1-1 是位级等价的）。P0-2 留到最后一并处理，它需要先定“以哪条路径为准”。

---

## 8. 被证否的设想

审计过程中提出、经实测否掉的判断，记录在此避免重复投入：

- **CUDA `torch.median` 有大小上限** —— 不成立。20.7 M 元素在 3060 上正常返回，`skin_gpu` 的全幅 MAD 不构成风险。
- **按列裁剪条带到人物 bbox** —— 无收益。该样本皮肤横向覆盖 98.5% 帧宽。
- **`np.clip(x+0.5)` 比 `round()` 省时** —— 相反。160.8 ms 变 239.8 ms，输出一致。
- **`astype(np.float32)*(1/255)` 比 `/255` 省时** —— 40.3 ms 对 39.7 ms，在噪声内。

---

## 9. 复现方法

诊断脚本已在审计结束后删除。复现关键数字的最小骨架：

```python
import os
os.environ["TMP"] = r"E:\SkinRetouch\tests\output"   # torch 导入需要可写临时目录
import cv2, numpy as np, sys
sys.path.insert(0, r"E:\SkinRetouch")
import skin_core

bgr = cv2.imdecode(np.fromfile(r"E:\SkinRetouch\tests\assets\DSC_7300.JPG", np.uint8),
                   cv2.IMREAD_COLOR)
out, mask, info = skin_core.skin_retouch(
    bgr, strength=0.75, texture_keep=0.35, blemish=0.7, tone_even=0.5,
    feather=0.5, conf=0.25, include_body=True, device="auto")
print(info)          # 分阶段耗时在这里
```

注意：不设 `TMP` 会得到 `FileNotFoundError: No usable temporary directory found`（`dill` 在 `import torch` 时就要用它）。

---

## 10. 处置结论（复核后追加）

逐条重新验证过。**静态代码发现全部成立**；**耗时数字不可采信**；一处归因过度。

### 10.1 对数字的复核

| 项目 | 本审计 | 复核实测 | 判定 |
|---|---|---|---|
| 成图逐像素最大差 | 76 灰阶 | 73~76 灰阶 | 吻合 |
| 成图逐像素平均差 | 0.96 | 0.952~0.961 | 吻合 |
| `faces` 阶段 | 205.5 ms | 186.6~204.7 ms | 吻合 |
| `person` 阶段 | 818.5 ms | 798.3~865.4 ms | 吻合 |
| `mask` 阶段 | 996.2 ms | 536.9~866.3 ms | **不吻合，波动过大** |
| `body` 阶段 | 770.1 ms | 543.4~848.1 ms | **不吻合，波动过大** |
| CPU/GPU 覆盖率相对差 | 2.07% | 1.44% | **不吻合** |
| 掩膜差 >32 占比 | 1.6% | 0.25% | **不吻合，差 6 倍** |

`mask` / `body` 两段在多次热态运行之间本身就有 ±300 ms 的抖动，单次采样不能立"偏差 34%"这种结论。掩膜统计那两项则是直接对不上。

### 10.2 P0-1 成立，但不是那 76 灰阶的主因

色度半径语义确实差一倍（`cv2.blur` 接核宽，`box_blur` 接半径），已修（`skin_gpu.py` `br = max(1, r_ch // 2)`）。修复前后实测：

| | 修前 | 修后 |
|---|---|---|
| 成图最大差 | 76 | 73 |
| 成图平均差 | 0.961 | 0.952 |
| p99 | 13.0 | 13.0 |

**几乎没动。** 主因是掩膜本身——CPU 用椭圆结构元、GPU 用矩形（torch 没有椭圆核），掩膜边界差 153 灰阶，边界像素一处被处理一处不被处理。这是 GPU 化已经付过的代价，不是这条 bug 造成的。原文"这是 76 灰阶尖峰的主因之一"应改为"次要因素之一"。

### 10.3 处置表

| 条目 | 判定 | 处置 |
|---|---|---|
| P0-1 色度半径 | 成立，非主因 | **已修** `skin_gpu.py` |
| P0-2 tau 两套算法 | 成立 | **不改代码**。两条路径的 ROI 划分本就不同，对齐要引入条带接缝。改为在 `HANDOFF.md` 写明两条路径不可交叉比较，并由 `compare_paths.py` 把差距钉住 |
| P0-3 偶数核崩溃 | 成立 | **已修** `_dilate`/`_erode` 强制 `\| 1` |
| P0-4 基线绑 CPU | 成立 | **已修文档** + 新增 `tests/compare_paths.py` 覆盖 GPU 路径 |
| P1-1 填洞 O(K·H·W) | 成立 | **已修** 改 `np.isin`，实测 140→98 ms 且位级一致。常态只有 9 个标签，收益集中在最坏情况 |
| P1-2 `skin_body` 无保护 | 成立 | **已修** try/except + `RuntimeWarning`，已实测缺模块时降级成功 |
| P1-3 静默降级 | 成立 | **已修** 两处 `except` 加 `warnings.warn(..., RuntimeWarning)` |
| P2-1 CPU 路径不优化 | 判断合理 | 未动 |
| P2-2 条带裁剪无收益 | 判断合理 | 未动 |
| P2-3 节点侧转换保持现状 | 判断合理 | 未动 |
| P3-1 零断言 | 成立 | **部分解决** `blemish_test.py` 加断言（`BASELINE OK` / 越界 `SystemExit(1)`），新增 `compare_paths.py`。未引入 pytest |
| P3-2 `bench_gpu.py` 不能当证据 | 成立 | **已修** 直接复用 `skin_gpu._box_mean` / `guided_filter`，删掉本地重复实现。重测：guided r=8 → cv2 388 ms / torch 46 ms |
| P3-3 文档偏差 | 成立 | **已修** `mask`/`body`/`person` 数字换成双图热态表；补 `device` 双默认；`detect_primary_person` 写明"仅面积最大"；`skin_body` 可缺省已由代码兑现；文件清单补 `compare_paths.py`，`probe_*.py` 实为 17 个（其中 2 个在验收链路上，15 个是失败路线取证） |
| P4 清洁度 | 前四条成立 | **已修** 删自拷贝、删 `probe` 死代码、`load_face_detector` 加 `FileNotFoundError`、`_reference_skin` 文档改为"取样块像素合并取中位数"。两套 `_keep_cuda_visible` / `_warm_cuda` **未合并**：跨模块抽公共函数的风险大于收益，两处各 8 行、语义已一致 |
| 第 8 节被证否项 | 复核一致 | 未重复投入 |

### 10.4 回归

改完复跑，`blemish_test.py` 三组基线逐位不变（基线走 CPU，只受 P1-1 影响，而 P1-1 位级等价）：

```
t01 clean=5.9 blur=1.3 ours=7.5 coverage=0.320
t07 clean=6.1 blur=2.0 ours=9.4 coverage=0.264
t05 clean=7.4 blur=2.0 ours=8.3 coverage=0.011
BASELINE OK (3 cases)
NODE OK
PATHS OK (rel 1.44%, p99 13.0)
```
