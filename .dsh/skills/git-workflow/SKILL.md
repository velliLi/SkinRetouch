---
name: git-workflow
description: '本项目（SkinRetouch，ComfyUI 保边美肤节点）的 git 操作规范：提交前验收门禁、提交信息格式、哪些资产不入库、远程推送与打 tag 发版。在提交、推送、开分支、发版、调整 .gitignore 或排查推送失败时使用。'
---

# SkinRetouch 的 git 管理

本项目是**单目录 ComfyUI 自定义节点**，无包管理、无构建步骤、无测试框架，git 是唯一的版本与发布通道。改动前先读 `HANDOFF.md` 的「不要推翻的决定」，那里记着踩过坑的算法决策。

## 仓库事实

| 项 | 值 |
|---|---|
| 项目根 | `E:\SkinRetouch`（含 `.git`，skill 扫描根也据此解析） |
| 远程 | `origin` = `git@github.com:velliLi/SkinRetouch.git`（**SSH**，HTTPS 在本机不可用，见「环境坑」） |
| 主分支 | `main`，本地与远程同名，`git push` 直接推 `main` |
| 身份 | **仓库级** `user.name=velliLi` / `user.email=vellichoron@gmail.com`，不是全局配置 |
| ComfyUI 挂载 | `E:\comfyui\ComfyUI\custom_nodes\ComfyUI-SkinRetouch` 是指向本目录的 **Junction** |
| 版本声明 | 代码里**没有** `__version__`、`pyproject.toml` 或 `setup.py`；版本只由 git tag 表达 |

在新环境 clone 后第一件事：

```powershell
git -C E:\SkinRetouch config user.name "velliLi"
git -C E:\SkinRetouch config user.email "vellichoron@gmail.com"
```

**永远在 `E:\SkinRetouch` 里执行 git**，不要在 Junction 路径下操作——两个路径指向同一份 `.git`，混用会让状态输出和 IDE 提示互相打架。

## 提交前必过验收门禁

算法改动不跑基线就提交，等于把回归埋进历史。三条命令，按顺序：

```powershell
$env:TMP="E:\comfyui\ComfyUI\temp"; $env:TEMP="E:\comfyui\ComfyUI\temp"
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\tests\blemish_test.py
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\tests\node_test.py
& E:\comfyui\python_embeded\python.exe E:\SkinRetouch\tests\compare_paths.py
```

期望输出：

- `blemish_test.py` 打印 `DONE`，且三行 `hf-energy` 与下面的基线**逐位一致**（数值越界会 `SystemExit(1)` 并打印 `FAIL`）：
  ```
  t01 hf-energy clean=5.9 blur=1.3 ours=7.5 coverage=0.320
  t07 hf-energy clean=6.1 blur=2.0 ours=9.4 coverage=0.264
  t05 hf-energy clean=7.4 blur=2.0 ours=8.3 coverage=0.011
  ```
- `node_test.py` 打印 `NODE OK`。
- `compare_paths.py` 打印通过：CPU/GPU 覆盖率相对差 <3%、成图 p99 <15、均值差 <2。

这三条依赖本机的 `python_embeded`（Python 3.11.9）、ultralytics、torch+CUDA 和 `models/` 权重，**CI 里跑不起来**，只能在本机执行。GitHub Actions 最多做语法门禁：

```powershell
& E:\comfyui\python_embeded\python.exe -m compileall -q E:\SkinRetouch\skin_core.py E:\SkinRetouch\skin_body.py E:\SkinRetouch\skin_gpu.py E:\SkinRetouch\skin_retouch_node.py
```

**只改文档**（`README.md`、`HANDOFF.md`、`OPTIMIZATION_AUDIT.md`）或本 skill 时，跳过算法门禁；**动了 `skin_core.py` / `skin_gpu.py` 中的滤波、掩膜、羽化、cumsum 算法**时，三条全跑，且基线数值变了必须在提交正文里说明为什么变、变了多少。

## 提交信息格式

标题用中文，一句话说清改了什么，不带 `feat:`/`fix:` 前缀；主体用 `-` 列表逐个点出改动位置，算法类改动附实测数字。

```
Skin Retouch (Lite): ComfyUI 保边美肤节点

- skin_core: 频域分离缺陷与结构边缘，自适应肤色模型约束处理范围
- skin_body: YOLOv8-seg 定位主要人物，覆盖面/颈/肩/手臂/手/胸口
```

```
添加 .gitattributes 统一换行规范
```

一次提交只做一件事。算法改动与文档整理不要混在一个提交里——`git bisect` 在 20 MP 管线上的代价很高，历史越干净越省事。

## 哪些资产不入库

`.gitignore` 已固化这些规则，不要为了"方便"放开：

| 路径 | 体积 | 不入库原因 |
|---|---|---|
| `models/*.pt` `*.onnx` `*.pth` | 102 MB | `face_yolov8m.pt` 50 MB + `person_yolov8m-seg.pt` 55 MB，是运行环境依赖；单文件已过 GitHub 50 MB 警告线 |
| `tests/assets/` | 22 MB | 含相机原图真人照（`DSC_4081.JPG`、`DSC_7300.JPG`），公开仓库不带真人素材 |
| `tests/output/` | 74 MB | 对照图产物，可由脚本重生成 |
| `.yolo/` | — | ultralytics 运行时缓存，写入本机绝对路径 |
| `__pycache__/` | — | 字节码 |

**注意冲突**：`HANDOFF.md` 的「环境事实」一节写着模型"随项目自带"，那是本机视角；仓库视角是**不入库**。新 clone 后必须手动补齐，否则 `blemish_test.py` 直接报找不到检测器：

```powershell
Copy-Item E:\comfyui\ComfyUI\models\ultralytics\bbox\face_yolov8m.pt E:\SkinRetouch\models\
Copy-Item E:\comfyui\ComfyUI\models\ultralytics\segm\person_yolov8m-seg.pt E:\SkinRetouch\models\
```

`.gitattributes` 设了 `* text=auto eol=lf`，源码按 LF 入库。如果某次 `git status` 突然冒出大量文件全是换行差异，别急着提交，先 `git add --renormalize .` 再 diff。

## 分支与发版

单人项目，默认直推 `main`。只有两种情况开分支：**要连续几天的算法重构**（`git switch -c algo/<主题>`，验收全绿后再合回），或者**试一条可能推翻的路线**——本项目已有 15 个 `tests/probe_*.py` 是失败路线的取证，留下的意义就是不再重复踩，分支同理。

发版靠 tag，本项目没有代码内版本号：

```powershell
git -C E:\SkinRetouch tag -a v0.1.0 -m "首个可用版本：脸 + 身体皮肤，CPU/GPU 双路径"
git -C E:\SkinRetouch push origin --tags
```

打 tag 前必须满足：三条基线全绿、`README.md` 的依赖表与实际一致。tag 与那三行 `hf-energy` 是绑定的，历史 tag 对应的基线数值一旦查不到，回滚就没有判据。

## 环境坑（都在本机验证过）

**HTTPS 到 GitHub 会假失败。** `curl https://github.com` 返回 `000`、报 `schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS`，看起来像网络不通，实际 TCP 早连上了——是 curl 的 Schannel 后端拿不到凭据。走 OpenSSL 栈的 Python 访问 `api.github.com` 正常返 200。需要调 GitHub API 时用 Python `urllib`/`requests`，别在 curl 上浪费时间。

**`git push` 在沙箱里会被管道权限挡住。** 报 `cannot create standard input pipe for ssh: Permission denied` → `fatal: unable to fork`。这是流程限制，不是密钥问题（`ssh -T git@github.com` 能正常返回 `Hi velliLi!`）。换一个不受该限制的终端或提权后再推。

**`winget install` 在本机返回 `0x8A15`**，装不了 `gh`。远程建仓库走 OAuth 设备流（`https://github.com/login/device` 输入一次性码），别指望 `gh repo create`。

**GitHub 单文件硬限 100 MB、警告线 50 MB。** 任何往仓库里加二进制的念头，先算体积。

**凭据不落库。** token 只写 `%TEMP%`，用完即删；永远不要 `git add` 含 token、密钥、本机绝对路径凭据的文件。`tests/probe_*.py` 里的 `E:\comfyui\...` 绝对路径是既有事实，不属于凭据泄露，但不鼓励再新增。

## 当前状态速查

```powershell
git -C E:\SkinRetouch log --oneline -5
git -C E:\SkinRetouch status -sb
git -C E:\SkinRetouch diff origin/main --stat
```

`## main...origin/main` 后面没有 `[ahead N]` 就说明远程同步。
