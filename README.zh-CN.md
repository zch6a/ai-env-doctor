<div align="center">

# ai-env-doctor

**为什么 PyTorch 用不了我的显卡？** 一条命令检查显卡、驱动、你实际装的 PyTorch、代理和镜像，告诉你哪里有问题，并给出能直接复制的修复命令。然后列出这台电脑能跑的本地模型。

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

在一台用 `pip install torch` 默认方式装了 PyTorch 的笔记本上：

```text
检查结果:
  ✗ C:\proj\.venv\Scripts\python.exe 里的 PyTorch 2.15.0+cpu 是纯 CPU 版：用不了 NVIDIA GeForce RTX 3060 Laptop GPU
      这是 torch.cuda.is_available() 为 False 最常见的原因：不指定 PyTorch 自己的源直接 `pip install torch`，在 Windows 上装到的就是 CPU 版（uv 在任何系统上都可能选中它）。
      解决: C:\proj\.venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  · 最新的 PyTorch（2.15.0）需要支持 CUDA 13.2 的驱动；当前驱动只支持 CUDA 12.7
      这个驱动能跑的最新 PyTorch 是 2.14.1（CUDA 12.6，cu126）；这里给出的命令装的就是它。
      解决: 要用 PyTorch 2.15.0，请更新驱动：https://www.nvidia.com/drivers
  ✓ NVIDIA 显卡：NVIDIA GeForce RTX 3060 Laptop GPU（6.0 GB），驱动 566.36，最高支持 CUDA 12.7
```

注意第二条：最直接的修法“装 CUDA 版”会装上 PyTorch 2.15，而它只有 CUDA 13 的版本，在这个驱动上同样用不了。ai-env-doctor 会读取 PyTorch 自己的软件源，知道到底有哪些版本，再挑出这个驱动能跑的最新那个。

## 检查什么

- **显卡和驱动**：NVIDIA（通过 `nvidia-smi`）、Apple 芯片、AMD 和其他（按操作系统列出的显示适配器）。有显卡没驱动、更新驱动后没重启导致驱动没加载、WSL 里没有 Windows 驱动、驱动太旧跑不了现在的 PyTorch。
- **你实际会用到的那个 PyTorch**：依次检查项目的 `.venv`、当前激活的虚拟环境或 conda 环境、PATH 上的 `python`（也可以用 `--python 路径` 指定）。它在每个环境自己的进程里去问，所以看到的就是你的代码看到的：
  - 有 NVIDIA 显卡却装了纯 CPU 版，这是最常见的问题；
  - 装的版本比驱动支持的 CUDA 新（CUDA 13 的 PyTorch 配 CUDA 12 的驱动）；
  - 版本里没有为这块显卡编译的代码（RTX 50 系列需要 CUDA 12.8 或更高）；
  - CUDA 启动失败，并附上 PyTorch 的原话；NVIDIA 显卡装了 AMD 版；
  - Apple 显卡（MPS）支持，以及在 Apple 芯片上用 Rosetta 跑的 Python；
  - 两个环境不一致（一个能用显卡，编辑器用的那个却不能）；
  - 32 位 Python、微软商店的 `python` 占位程序、导入失败的 PyTorch（并指出缺的 DLL）、和 PyTorch 不配套的 torchvision、在 Windows 上太旧不支持显卡的 bitsandbytes。
- **代理和证书**：只设置了 HTTP 代理没有 HTTPS 代理（所有下载都绕过了代理）、设置了代理却没有在 NO_PROXY 里排除 localhost（Ollama 这类本机服务会出错）、SOCKS 代理、指定的证书文件不存在、关闭了证书校验。
- **网络**：pypi.org、download.pytorch.org、huggingface.co、hf-mirror.com 和 Ollama 的模型仓库能不能通过你的代理连上；区分 DNS、超时、拒绝连接和证书错误；Hugging Face 被挡而镜像能连时，建议使用镜像。
- **放模型的空间**：Hugging Face 和 Ollama 存模型的磁盘还剩多少；Ollama 有没有在运行。
- **这台电脑能跑的模型**：对话、写代码、推理、看图、向量这几类，各自能完整放进显存（没有显卡时放进内存）的最大常用模型，以及大一号、由显卡和 CPU 分担运行的模型。完整的模型目录和速度估计见 [llmfit](https://github.com/AlexsJones/llmfit)。

## 放心运行

- **只读。** 不安装、不删除、不修改任何东西；修复命令只是打印出来，由你自己决定是否运行。
- **不发送任何关于你的信息。** 网络检查只是向上面五个地址各发一个 HEAD 请求（和 pip 一样走你的代理），再读取一次 PyTorch 公开的软件源列表；请求里不带任何关于这台电脑的信息。加 `--offline` 就全部跳过。
- **代理地址里的密码会被隐藏**，无论是报告还是 `--json`，都可以直接贴到问题反馈里。
- 没有任何依赖：只是 Python 3.10+ 的标准库，所以 `uvx` 一两秒就能启动。

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

在项目文件夹里运行：它会先检查那里的 `.venv`。有需要修复的问题时退出码为 1，否则为 0。

## 状态

0.1，开发中。用各种假的 PyTorch 版本（CPU 版、CUDA 正常、CUDA 启动失败、缺少显卡代码、导入失败）测试真实的探测程序，用合成的机器测试每条规则，用本地的模拟服务测试网络检查。在 Windows 11、RTX 3060 笔记本显卡（驱动支持 CUDA 12.7）上手动试过：用 `pip install torch` 默认方式装的 PyTorch 被正确诊断为 CPU 版，如上所示。macOS、Linux 和 WSL 由测试和 CI 覆盖，还没有手动试过。

## 开发

```bash
uv venv && uv pip install -e . pytest ruff
uv run pytest
```

## 许可证

[MIT](LICENSE)
