<div align="center">

# ai-env-doctor

**为什么 PyTorch 用不了我的显卡？** 一条命令检查显卡和驱动、你实际装的 PyTorch，以及代理和镜像，告诉你哪里有问题，并给出能直接复制的修复命令。然后列出这台电脑能跑的本地模型。

<a href="https://github.com/zch6a/ai-env-doctor/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/zch6a/ai-env-doctor/actions/workflows/ci.yml/badge.svg"></a>
<img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white">
<img alt="无依赖" src="https://img.shields.io/badge/dependencies-none-2ea44f">
<img alt="只读" src="https://img.shields.io/badge/changes%20nothing-read--only-2ea44f">
<a href="LICENSE"><img alt="MIT 许可证" src="https://img.shields.io/badge/license-MIT-blue"></a>

[English](README.md) · **简体中文**

</div>

---

```bash
uvx --from git+https://github.com/zch6a/ai-env-doctor ai-env-doctor
```

在一台用 `uv pip install torch` 默认方式装了 PyTorch 的笔记本上：

```text
检查结果:
  ✗ C:\proj\.venv\Scripts\python.exe 里的 PyTorch 2.15.0+cpu 是纯 CPU 版：用不了 NVIDIA GeForce RTX 3060 Laptop GPU
      这是 torch.cuda.is_available() 为 False 最常见的原因：不指定 PyTorch 自己的源直接 `pip install torch`，在 Windows 上装到的就是 CPU 版（Linux 上 PyPI 的 torch 是 CUDA 版：那里的 CPU 版来自 PyTorch 的 CPU 源，或者锁文件选了它）。
      解决: uv pip uninstall --python C:\proj\.venv\Scripts\python.exe torch
           uv pip install --python C:\proj\.venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu126
  · 最新的 PyTorch（2.15.0）需要支持 CUDA 13.2 的驱动；当前驱动只支持 CUDA 12.7
      这个驱动能跑的最新 PyTorch 是 2.14.1（CUDA 12.6，cu126）；这里给出的命令装的就是它。
      解决: 要用 PyTorch 2.15.0，请更新驱动：https://www.nvidia.com/drivers
  ✓ NVIDIA 显卡：NVIDIA GeForce RTX 3060 Laptop GPU（6.0 GB），驱动 566.36，最高支持 CUDA 12.7
```

运行这两条命令后装上了 PyTorch 2.14.1+cu126，能在显卡上运行，ai-env-doctor 随后显示 `✓ PyTorch 2.14.1+cu126 能使用显卡`。注意第二条：最直接的修法“装 CUDA 版”会装上 PyTorch 2.15，而它只有 CUDA 13 的版本，在这个驱动上同样用不了。ai-env-doctor 会读取 PyTorch 自己的软件源，知道到底有哪些版本，再挑出这个驱动能跑的最新那个。这里的命令用的是 `uv pip`，因为这个环境是 uv 建的、没有 pip；其他情况下是 `python -m pip`。

## 检查什么

- **显卡和驱动**：NVIDIA（通过 `nvidia-smi`）、Apple 芯片、AMD 和其他（按操作系统列出的显示适配器；在 WSL 里按 Windows 列出的）。有显卡没驱动、更新驱动后没重启导致驱动没加载、有驱动但没有 `nvidia-smi`、WSL 里没有 Windows 驱动（以及完全用不了显卡的 WSL 1）、容器启动时没有传入显卡、驱动太旧跑不了现在的 PyTorch。
- **你实际会用到的那个 PyTorch**：依次检查项目的 `.venv`、当前激活的虚拟环境或 conda 环境、PATH 上的 `python`（也可以用 `--python 路径` 指定）。它在每个环境自己的进程里去问，所以看到的就是你的代码看到的：
  - 有 NVIDIA 显卡却装了纯 CPU 版，这是最常见的问题；
  - 装的版本比驱动支持的 CUDA 新（CUDA 13 的 PyTorch 配 CUDA 12 的驱动）；
  - 版本里没有为这块显卡编译的代码（RTX 50 系列需要 CUDA 12.8 或更高）；
  - CUDA 启动失败，并附上 PyTorch 的原话；NVIDIA 显卡装了 AMD 版；
  - AMD 显卡上，PyTorch 是不是能看到显卡的 ROCm 版，以及该从哪里装（Linux 上是 PyTorch 的源，Windows 和 WSL 上是 AMD 自己的）；
  - Apple 显卡（MPS）支持，以及在 Apple 芯片上用 Rosetta 跑的 Python；
  - 两个环境不一致（一个能用显卡，编辑器用的那个却不能）；
  - conda 环境里由 conda 装的 PyTorch（pip 删不掉）和 `cpuonly`（它会让 CPU 版一装再装）：修复命令会先用 conda 删掉它们；
  - pip 不允许往里装包的系统 Python（PEP 668：修复命令会新建虚拟环境）；
  - 32 位 Python、微软商店的 `python` 占位程序、导入失败的 PyTorch（并指出缺的 DLL）、和 PyTorch 不配套的 torchvision、在 Windows 上太旧不支持显卡的 bitsandbytes。
- **代理和证书**：只设置了 HTTP 代理没有 HTTPS 代理（所有下载都绕过了代理）、设置了代理却没有在 NO_PROXY 里排除 localhost（Ollama 这类本机服务会出错）、SOCKS 代理、指定的证书文件不存在、关闭了证书校验。
- **网络**：pypi.org、download.pytorch.org、huggingface.co、hf-mirror.com 和 Ollama 的模型仓库能不能通过你的代理连上；区分 DNS、超时、拒绝连接和证书错误；Hugging Face 被挡而镜像能连时，建议使用镜像。
- **放模型的空间**：Hugging Face 和 Ollama 存模型的磁盘还剩多少；本机的 Ollama 有没有在运行（只向 localhost 发一个请求，从不经过代理）。
- **这台电脑能跑的模型**：对话、写代码、推理、看图、向量这几类，各自能完整放进显存（没有显卡时放进内存）的最大常用模型，以及大一号、由显卡和 CPU 分担运行的模型。这份清单注明了日期（2026 年 11 月），过时了报告会提醒。完整的模型目录和速度估计见 [llmfit](https://github.com/AlexsJones/llmfit)。

## 放心运行

- **不改动任何东西。** 不安装、不删除、不写入任何东西；修复命令只是打印出来，由你自己决定是否运行。它执行的只有：系统自带的硬件查询工具（`nvidia-smi`、`lspci`、PowerShell 的 `Get-CimInstance`、`sysctl`），以及它检查的每个 Python（项目的 `.venv`、当前激活的环境、PATH 上的 `python`），在里面执行一小段导入 torch 的脚本。这段脚本在一个无关的文件夹里运行，不会误用你项目里的文件。
- **不发送任何关于你的信息。** 网络检查只是向上面五个地址各发一个 HEAD 请求（和 pip 一样走你的代理），再读取一次 PyTorch 公开的软件源列表；请求里不带任何关于这台电脑的信息。加 `--offline` 就全部跳过。
- **代理地址里的密码会被隐藏**（用户名和 token 也是），报告和 `--json` 里都是。但路径不会隐藏，而路径里有你的用户名：贴出去之前看一眼。
- 没有任何依赖：只是 Python 3.10+ 的标准库。用 `uvx` 时第一次要下载并构建（这里约 15 秒），之后马上就能启动。

## 直接问 AI 助手

ai-env-doctor 自带一个 Agent Skill（技能）。在 Claude Code 里：

```text
/plugin marketplace add zch6a/ai-env-doctor
/plugin install ai-env-doctor@ai-env-doctor
```

然后问它“为什么 `torch.cuda.is_available()` 是 False？”“pip install torch 卡住了，哪里出了问题？”或者“我的显卡能跑哪个本地模型？”。助手会在你的项目里运行 ai-env-doctor，读它的 JSON 结果，用大白话讲清楚问题；修复命令会先给你看，经你同意才运行。从文件夹读取技能的其他助手：把 [`plugin/skills/ai-env-doctor`](plugin/skills/ai-env-doctor/SKILL.md) 复制到它们的技能文件夹。

## 安装和使用

```bash
uvx --from git+https://github.com/zch6a/ai-env-doctor ai-env-doctor      # 运行一次，不安装任何东西
uv tool install git+https://github.com/zch6a/ai-env-doctor               # 或者安装这个命令
pip install "ai-env-doctor @ git+https://github.com/zch6a/ai-env-doctor"
```

```bash
ai-env-doctor                    # 检查所有内容（报告语言跟随系统）
ai-env-doctor --lang zh          # 中文；--lang en 为英文
ai-env-doctor --python 路径      # 检查指定的解释器（可以给多次）
ai-env-doctor --offline          # 不检查网络
ai-env-doctor --json             # 输出所有检查结果，便于脚本处理和提交问题
```

在项目文件夹里运行：它会先检查那里的 `.venv`。有需要修复的问题时退出码为 1，否则为 0，工具自己出错时为 2。`--json` 每个字段的含义见 [docs/json.md](docs/json.md)（英文）。

## 状态

1.0。用各种假的 PyTorch 版本（CPU 版、CUDA 正常、CUDA 在启动时或运行内核时失败、缺少显卡代码、导入失败）测试真实的探测程序，用合成的机器测试每条规则，修复命令交给 PowerShell 和 bash 实际解析，用本地的模拟服务测试网络检查。由没写过它的人独立审查过两次（见 [POLISH_LOG.md](POLISH_LOG.md)）。在 Windows 11、RTX 3060 笔记本显卡（驱动支持 CUDA 12.7）上完整地手动试过：诊断出 CPU 版、运行打印出来的命令、之后显卡可以正常使用；也从 GitHub 用 `uvx` 实际运行过。CI 在 Linux（无显卡）和 macOS（Apple M1）上运行它。WSL、容器、由 conda 装了 PyTorch 的 conda 环境和 AMD 显卡只由测试覆盖：欢迎提交这些机器上的运行结果。

## 开发

```bash
uv venv && uv pip install -e . pytest ruff
uv run pytest
```

## 许可证

[MIT](LICENSE)
