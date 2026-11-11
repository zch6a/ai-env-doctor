"""What each finding says, in English and Chinese: a title, what it means, and the fix."""

from __future__ import annotations

import string

DRIVERS = "https://www.nvidia.com/drivers"
WHEEL_HELP = "https://pytorch.org/get-started/locally/"

M: dict[str, dict[str, tuple[str, str, str]]] = {
    # ---- GPU and driver
    "gpu-nvidia": {
        "en": ("NVIDIA GPU: {name} ({memory}), driver {driver}, supports CUDA up to {cuda}", "", ""),
        "zh": ("NVIDIA 显卡：{name}（{memory}），驱动 {driver}，最高支持 CUDA {cuda}", "", "")},
    "driver-old": {
        "en": ("The NVIDIA driver only supports CUDA {cuda}", "Current PyTorch builds need a driver for CUDA 12 or later.",
               f"Update the driver: {DRIVERS}"),
        "zh": ("NVIDIA 驱动只支持到 CUDA {cuda}", "现在的 PyTorch 需要支持 CUDA 12 或更高版本的驱动。", f"更新驱动：{DRIVERS}")},
    "nvidia-smi-failed": {
        "en": ("nvidia-smi is installed but cannot talk to the GPU", "The driver is not loaded, or was updated without a reboot "
               "(\"Driver/library version mismatch\").", "Restart the computer; if that does not help, reinstall the driver: " + DRIVERS),
        "zh": ("装了 nvidia-smi，但它连不上显卡", "驱动没有加载，或者更新驱动后还没重启（“Driver/library version mismatch”）。",
               "重启电脑；还不行就重装驱动：" + DRIVERS)},
    "driver-missing": {
        "en": ("{name} is there, but its NVIDIA driver is not", "Without the driver, nothing can use the GPU.",
               f"Install the driver: {DRIVERS}"),
        "zh": ("有 {name}，但没有装 NVIDIA 驱动", "没有驱动，任何程序都用不了这块显卡。", f"安装驱动：{DRIVERS}")},
    "wsl-no-driver": {
        "en": ("WSL cannot see the NVIDIA GPU ({name})", "In WSL the GPU comes from the Windows driver; a Linux NVIDIA driver must not be "
               "installed inside WSL.", f"Install or update the driver in Windows ({DRIVERS}), then run `wsl --update` and `wsl --shutdown`."),
        "zh": ("WSL 看不到 NVIDIA 显卡（{name}）", "WSL 里的显卡来自 Windows 的驱动；不要在 WSL 里面装 Linux 版的 NVIDIA 驱动。",
               f"在 Windows 里安装或更新驱动（{DRIVERS}），然后运行 `wsl --update` 和 `wsl --shutdown`。")},
    "gpu-apple": {
        "en": ("Apple GPU: {name}, about {memory} of memory it may use", "", ""),
        "zh": ("Apple 显卡：{name}，可用内存约 {memory}", "", "")},
    "gpu-amd": {
        "en": ("AMD GPU: {name}", "PyTorch uses AMD GPUs through ROCm, on Linux and some Windows setups; this tool does not check "
               "ROCm in detail yet.", "See https://rocm.docs.amd.com for what your GPU supports."),
        "zh": ("AMD 显卡：{name}", "PyTorch 通过 ROCm 使用 AMD 显卡（Linux，以及部分 Windows 环境）；本工具暂时不细查 ROCm。",
               "你的显卡支持什么，见 https://rocm.docs.amd.com")},
    "wsl-gpu-unknown": {
        "en": ("WSL can reach a GPU, but not through NVIDIA's driver", "WSL passes every Windows GPU through (/dev/dxg), and "
               "which one this is could not be read: running powershell.exe from WSL failed. Without nvidia-smi, PyTorch here "
               "runs on the CPU.", "If this PC has an NVIDIA GPU: update its driver in Windows (" + DRIVERS + "), then run "
               "`wsl --shutdown` in Windows. Do not install a Linux NVIDIA driver inside WSL."),
        "zh": ("WSL 能访问一块显卡，但不是通过 NVIDIA 驱动", "WSL 会把 Windows 的所有显卡都接进来（/dev/dxg），但读不出这块是什么显卡："
               "在 WSL 里运行 powershell.exe 失败了。没有 nvidia-smi，这里的 PyTorch 只能用 CPU。",
               "如果这台电脑有 NVIDIA 显卡：在 Windows 里更新驱动（" + DRIVERS + "），然后在 Windows 里运行 `wsl --shutdown`。"
               "不要在 WSL 里安装 Linux 版的 NVIDIA 驱动。")},
    "gpu-none": {
        "en": ("No GPU for AI work: everything runs on the CPU ({ram} of memory)", "Small models run well enough on a CPU "
               "(see the models below); training and large models will be slow.", ""),
        "zh": ("没有可用于 AI 的显卡：一切都在 CPU 上运行（内存 {ram}）", "小模型在 CPU 上也能用（见下面的模型）；训练和大模型会很慢。", "")},
    "rosetta": {
        "en": ("{path} is an Intel (x86_64) Python on Apple Silicon: it runs under Rosetta", "PyTorch then cannot use the Apple GPU (MPS) and runs "
               "several times slower.", "Install an arm64 Python (from python.org, Homebrew, or `uv python install`) and recreate the environment."),
        "zh": ("{path} 是 Intel（x86_64）版的 Python，在 Apple 芯片上运行于 Rosetta 之下", "这样 PyTorch 用不了 Apple 显卡（MPS），速度会慢好几倍。",
               "安装 arm64 版的 Python（python.org、Homebrew 或 `uv python install`），然后重建环境。")},
    "cuda-visible-devices": {
        "en": ("CUDA_VISIBLE_DEVICES={value} hides the GPUs", "Programs will see no GPU, or not the one you meant.",
               "Unset it (`unset CUDA_VISIBLE_DEVICES`, or `Remove-Item Env:CUDA_VISIBLE_DEVICES` in PowerShell), or set it to 0."),
        "zh": ("CUDA_VISIBLE_DEVICES={value} 把显卡藏起来了", "程序会看不到显卡，或者看到的不是你想用的那块。",
               "取消这个变量（`unset CUDA_VISIBLE_DEVICES`，PowerShell 里是 `Remove-Item Env:CUDA_VISIBLE_DEVICES`），或者设为 0。")},
    # ---- Python
    "no-python": {
        "en": ("No Python could be checked", "No .venv here, no active environment, and no working `python` on PATH, so nothing about PyTorch can be said.",
               "Install Python 3.10 or later, or pass the one you use with --python."),
        "zh": ("没有能检查的 Python", "这里没有 .venv，也没有激活的环境，PATH 上也没有能运行的 `python`，所以无法判断 PyTorch 的情况。", "安装 Python 3.10 或更高版本，或者用 --python 指定你用的那个。")},
    "store-stub": {
        "en": ("`python` on PATH is the Microsoft Store placeholder ({path})", "It opens the Store instead of running Python, so scripts "
               "that call `python` fail.", "Install Python (python.org or `winget install Python.Python.3.12`), or turn off the "
               "\"python.exe\" app execution alias in Settings > Apps > Advanced app settings."),
        "zh": ("PATH 上的 `python` 是微软商店的占位程序（{path}）", "它会打开商店而不是运行 Python，调用 `python` 的脚本都会失败。",
               "安装 Python（python.org 或 `winget install Python.Python.3.12`），或者在“设置 > 应用 > 高级应用设置 > 应用执行别名”里关掉 python.exe。")},
    "python-failed": {
        "en": ("Could not look inside {path}", "{error}", ""),
        "zh": ("无法检查 {path}", "{error}", "")},
    "python-32bit": {
        "en": ("{path} is a 32-bit Python", "PyTorch and most AI packages exist only for 64-bit Python.", "Install a 64-bit Python 3.10 or later."),
        "zh": ("{path} 是 32 位的 Python", "PyTorch 和大多数 AI 包只有 64 位版本。", "安装 64 位的 Python 3.10 或更高版本。")},
    "torch-missing": {
        "en": ("PyTorch is not installed in {why}", "Python {python} at {path}. If you need it there, this installs the build for your GPU:",
               "{fix_cmd}"),
        "zh": ("{why}里没有安装 PyTorch", "Python {python}，位于 {path}。如果要在这里用，下面的命令会装上适合你显卡的版本：", "{fix_cmd}")},
    "torch-import": {
        "en": ("PyTorch is installed in {path} but fails to import", "{error}",
               "Reinstall it in that environment (see " + WHEEL_HELP + "); a DLL error on Windows often means the Microsoft Visual C++ "
               "Redistributable is missing: https://aka.ms/vs/17/release/vc_redist.x64.exe"),
        "zh": ("{path} 里装了 PyTorch，但导入失败", "{error}",
               "在那个环境里重装（见 " + WHEEL_HELP + "）；Windows 上的 DLL 错误通常是缺少 Microsoft Visual C++ 运行库："
               "https://aka.ms/vs/17/release/vc_redist.x64.exe")},
    "mps-not-built": {
        "en": ("PyTorch {version} in {path} has no Apple GPU support", "This build cannot use MPS.",
               "Reinstall a standard macOS arm64 build: {fix_cmd}"),
        "zh": ("{path} 里的 PyTorch {version} 不支持 Apple 显卡", "这个版本用不了 MPS。",
               "重装标准的 macOS arm64 版：{fix_cmd}")},
    "mps-unavailable": {
        "en": ("PyTorch {version} supports MPS, but it is not available", "MPS needs macOS 12.3 or later on Apple Silicon.",
               "Update macOS, and make sure Python is not running under Rosetta."),
        "zh": ("PyTorch {version} 支持 MPS，但当前不可用", "MPS 需要 Apple 芯片和 macOS 12.3 或更高版本。", "升级 macOS，并确认 Python 不是在 Rosetta 下运行。")},
    "torch-mps": {
        "en": ("PyTorch {version} uses the Apple GPU (MPS)", "{path}", ""),
        "zh": ("PyTorch {version} 能使用 Apple 显卡（MPS）", "{path}", "")},
    "torch-rocm-on-nvidia": {
        "en": ("PyTorch {version} is the AMD (ROCm) build, but the GPU is NVIDIA", "It will never use this GPU.", "{fix_cmd}"),
        "zh": ("PyTorch {version} 是 AMD（ROCm）版，但显卡是 NVIDIA 的", "它永远用不了这块显卡。", "{fix_cmd}")},
    "torch-from-conda": {
        "en": ("In {path}, conda installed PyTorch ({packages}, from {channel})",
               "pip cannot remove packages conda installed, so the fix removes them with conda first; conda lists everything it will "
               "remove (packages that need PyTorch go too) and asks before doing it. Then pip installs PyTorch's own build, which is "
               "what PyTorch recommends inside conda. To stay with conda packages instead: `conda install -c conda-forge pytorch-gpu`.", ""),
        "zh": ("{path} 里的 PyTorch 是 conda 装的（{packages}，来自 {channel}）",
               "pip 删不掉 conda 装的包，所以修复命令先用 conda 删除它们；conda 会列出要删除的所有包（依赖 PyTorch 的包也会一起删）并先征求你的同意。"
               "然后用 pip 装 PyTorch 官方的版本，这也是 PyTorch 推荐在 conda 里使用的方式。如果想继续用 conda 的包：`conda install -c conda-forge pytorch-gpu`。", "")},
    "torch-from-conda-cpuonly": {
        "en": ("In {path}, conda installed PyTorch with cpuonly ({packages}, from {channel})",
               "cpuonly is what makes conda choose the CPU build, every time PyTorch is installed or updated. pip cannot remove packages "
               "conda installed, so the fix removes them with conda first; conda lists everything it will remove and asks before doing "
               "it. Then pip installs PyTorch's own build, which is what PyTorch recommends inside conda.", ""),
        "zh": ("{path} 里的 PyTorch 是 conda 装的，而且装了 cpuonly（{packages}，来自 {channel}）",
               "正是 cpuonly 让 conda 每次安装或更新 PyTorch 时都选 CPU 版。pip 删不掉 conda 装的包，所以修复命令先用 conda 删除它们；"
               "conda 会列出要删除的所有包并先征求你的同意。然后用 pip 装 PyTorch 官方的版本，这也是 PyTorch 推荐在 conda 里使用的方式。", "")},
    "torch-cpu": {
        "en": ("PyTorch {version} in {path} is the CPU-only build: it will not use the {gpu}",
               "This is the most common reason for torch.cuda.is_available() to be False: `pip install torch` without PyTorch's own "
               "index gives the CPU build on Windows (and uv may pick it anywhere).", "{fix_cmd}"),
        "zh": ("{path} 里的 PyTorch {version} 是纯 CPU 版：用不了 {gpu}",
               "这是 torch.cuda.is_available() 为 False 最常见的原因：不指定 PyTorch 自己的源直接 `pip install torch`，在 Windows 上装到的就是 CPU 版"
               "（uv 在任何系统上都可能选中它）。", "{fix_cmd}")},
    "wheel-major-newer": {
        "en": ("PyTorch {version} is built for CUDA {wheel}, but the driver only supports CUDA {driver}",
               "A new major CUDA version needs a new driver.", f"Update the driver ({DRIVERS}), or install a build this driver can run: {{fix_cmd}}"),
        "zh": ("PyTorch {version} 是为 CUDA {wheel} 编译的，但驱动只支持到 CUDA {driver}", "CUDA 大版本升级需要新驱动。",
               f"更新驱动（{DRIVERS}），或者装一个这个驱动能跑的版本：{{fix_cmd}}")},
    "arch-missing": {
        "en": ("PyTorch {version} has no code for the {gpu} (compute capability {cap})",
               "It was built for {archs}. RTX 50-series (Blackwell) GPUs need a build for CUDA 12.8 or later.", "{fix_cmd}"),
        "zh": ("PyTorch {version} 没有为 {gpu}（计算能力 {cap}）编译的代码", "它是为 {archs} 编译的。RTX 50 系列（Blackwell）显卡需要 CUDA 12.8 或更高版本的 PyTorch。",
               "{fix_cmd}")},
    "cuda-unavailable-minor": {
        "en": ("PyTorch {version} (CUDA {wheel}) cannot use the GPU; the driver supports CUDA {driver}",
               "A wheel for a newer CUDA 12 usually runs on an older CUDA 12 driver, but not always. PyTorch said: {reason}",
               f"Update the driver ({DRIVERS}), or install the build for this driver: {{fix_cmd}}"),
        "zh": ("PyTorch {version}（CUDA {wheel}）用不了显卡；驱动支持到 CUDA {driver}",
               "为较新的 CUDA 12 编译的版本通常能在较旧的 CUDA 12 驱动上运行，但不总是。PyTorch 的提示：{reason}",
               f"更新驱动（{DRIVERS}），或者装适合这个驱动的版本：{{fix_cmd}}")},
    "cuda-unavailable": {
        "en": ("PyTorch {version} is built for CUDA {wheel}, but CUDA does not start", "PyTorch said: {reason}",
               "Restart the computer if the driver was updated; check CUDA_VISIBLE_DEVICES; in WSL, run `wsl --shutdown`. "
               "If it persists, reinstall: {fix_cmd}"),
        "zh": ("PyTorch {version} 是 CUDA {wheel} 版，但 CUDA 启动不了", "PyTorch 的提示：{reason}",
               "如果刚更新过驱动，重启电脑；检查 CUDA_VISIBLE_DEVICES；WSL 里运行 `wsl --shutdown`。还不行就重装：{fix_cmd}")},
    "torch-cuda": {
        "en": ("PyTorch {version} uses the GPU ({gpu}, CUDA {wheel})", "{path}", ""),
        "zh": ("PyTorch {version} 能使用显卡（{gpu}，CUDA {wheel}）", "{path}", "")},
    "newest-needs-driver": {
        "en": ("The newest PyTorch ({newest}) needs a driver for {needs}; this one supports CUDA {driver}",
               "The newest PyTorch this driver can run is {best}, for CUDA {tag_cuda} ({tag}); the commands here install that.",
               f"To use PyTorch {{newest}}, update the driver: {DRIVERS}"),
        "zh": ("最新的 PyTorch（{newest}）需要支持 {needs} 的驱动；当前驱动只支持 CUDA {driver}",
               "这个驱动能跑的最新 PyTorch 是 {best}（CUDA {tag_cuda}，{tag}）；这里给出的命令装的就是它。", f"要用 PyTorch {{newest}}，请更新驱动：{DRIVERS}")},
    "driver-too-old-for-gpu": {
        "en": ("The driver (CUDA {driver}) is too old for any PyTorch that supports the {gpu}", "Update the driver first; then install PyTorch.",
               f"Update the driver: {DRIVERS}"),
        "zh": ("驱动（CUDA {driver}）太旧，没有任何支持 {gpu} 的 PyTorch 能在上面运行", "先更新驱动，再安装 PyTorch。", f"更新驱动：{DRIVERS}")},
    "wheel-minor-newer": {
        "en": ("The PyTorch build (CUDA {wheel}) is newer than the driver (CUDA {driver}); it works now",
               "Some features (compiling new kernels at run time) may need the newer driver.", f"Updating the driver avoids surprises: {DRIVERS}"),
        "zh": ("PyTorch 版本（CUDA {wheel}）比驱动（CUDA {driver}）新；目前能用", "部分功能（运行时编译新内核）可能需要新驱动。", f"更新驱动可以避免意外：{DRIVERS}")},
    "torch-rocm": {
        "en": ("PyTorch {version} uses the AMD GPU ({gpu}, ROCm)", "", ""),
        "zh": ("PyTorch {version} 能使用 AMD 显卡（{gpu}，ROCm）", "", "")},
    "rocm-unavailable": {
        "en": ("PyTorch {version} is the ROCm build, but it cannot use the GPU", "PyTorch said: {reason}",
               "Check that this GPU is supported by this ROCm version: https://rocm.docs.amd.com"),
        "zh": ("PyTorch {version} 是 ROCm 版，但用不了显卡", "PyTorch 的提示：{reason}", "确认这块显卡受这个 ROCm 版本支持：https://rocm.docs.amd.com")},
    "torch-cuda-no-gpu": {
        "en": ("PyTorch {version} is a CUDA {wheel} build, but there is no NVIDIA GPU", "It runs on the CPU, with a larger download than needed.", ""),
        "zh": ("PyTorch {version} 是 CUDA {wheel} 版，但这台电脑没有 NVIDIA 显卡", "它会在 CPU 上运行，只是安装包比需要的大。", "")},
    "torch-cpu-ok": {
        "en": ("PyTorch {version} runs on the CPU", "{path}", ""),
        "zh": ("PyTorch {version} 在 CPU 上运行", "{path}", "")},
    "environments-disagree": {
        "en": ("Your environments disagree: PyTorch uses the GPU in one, not in another",
               "GPU: {gpu}\nCPU only: {cpu}\nMake sure you run your code with the first one (activate it, or select it in your editor).", ""),
        "zh": ("你的几个环境不一致：一个能用显卡，另一个不能", "能用显卡：{gpu}\n只能用 CPU：{cpu}\n确认运行代码时用的是前一个（激活它，或者在编辑器里选中它）。", "")},
    "bnb-windows-old": {
        "en": ("bitsandbytes {version} has no GPU support on Windows", "Versions before 0.43 work on Windows only on the CPU.",
               "pip install -U bitsandbytes"),
        "zh": ("bitsandbytes {version} 在 Windows 上不支持显卡", "0.43 之前的版本在 Windows 上只能用 CPU。", "pip install -U bitsandbytes")},
    "companion-build-differs": {
        "en": ("{name} ({other}) was built for a different PyTorch than {torch}", "Mismatched builds fail with errors about missing "
               "operators or DLLs.", "Reinstall them together, from the same index, in one command."),
        "zh": ("{name}（{other}）和 PyTorch（{torch}）不是配套的版本", "不配套时会报找不到算子或 DLL 之类的错误。", "用同一个源、一条命令一起重装它们。")},
    # ---- proxies and certificates
    "proxy-no-scheme": {
        "en": ("{detail} has no http:// in front", "Some tools (pip among them) then misread it.", "Write it as http://host:port."),
        "zh": ("{detail} 前面没有 http://", "有些工具（包括 pip）会读错。", "写成 http://主机:端口。")},
    "proxy-bad-scheme": {
        "en": ("Unusual proxy scheme: {detail}", "Most tools understand http:// proxies only.", "Use an http:// address for the proxy."),
        "zh": ("少见的代理协议：{detail}", "大多数工具只认 http:// 代理。", "改用 http:// 开头的代理地址。")},
    "proxy-no-host": {
        "en": ("{detail} has no host", "Every request through it fails.", "Fix or unset the variable."),
        "zh": ("{detail} 里没有主机名", "通过它的请求都会失败。", "改正或取消这个变量。")},
    "proxy-socks": {
        "en": ("{detail} is a SOCKS proxy", "pip, huggingface_hub and urllib do not speak SOCKS without extra packages; downloads fail "
               "with odd errors.", "Use the proxy's HTTP port (most proxy apps offer one), or install the SOCKS extras (pip install \"requests[socks]\")."),
        "zh": ("{detail} 是 SOCKS 代理", "pip、huggingface_hub 和 urllib 不装额外的包就不支持 SOCKS，下载会报奇怪的错误。",
               "改用代理软件的 HTTP 端口（大多数代理软件都有），或者安装 SOCKS 支持（pip install \"requests[socks]\"）。")},
    "proxy-https-missing": {
        "en": ("Only an HTTP proxy is set ({detail}), no HTTPS_PROXY", "Every package and model download is HTTPS, so they all go around "
               "the proxy.", "Set HTTPS_PROXY to the same address."),
        "zh": ("只设置了 HTTP 代理（{detail}），没有 HTTPS_PROXY", "包和模型的下载都是 HTTPS，所以全都绕过了代理。", "把 HTTPS_PROXY 设成同一个地址。")},
    "proxy-localhost": {
        "en": ("A proxy is set, and NO_PROXY does not exempt localhost", "Requests to local servers (Ollama, LM Studio, vLLM, a notebook) "
               "may go to the proxy and fail.", "Set NO_PROXY=localhost,127.0.0.1,::1"),
        "zh": ("设置了代理，但 NO_PROXY 没有排除 localhost", "发给本机服务（Ollama、LM Studio、vLLM、Jupyter）的请求可能被发到代理而失败。",
               "设置 NO_PROXY=localhost,127.0.0.1,::1")},
    "ca-missing": {
        "en": ("A certificate bundle is set but the file is not there: {detail}", "Every HTTPS download will fail its certificate check.",
               "Point it at your organization's CA file, or unset it."),
        "zh": ("设置了证书文件，但文件不存在：{detail}", "所有 HTTPS 下载都会证书校验失败。", "指向你们单位的 CA 证书文件，或者取消这个变量。")},
    "hf-ssl-off": {
        "en": ("Certificate checks for Hugging Face are turned off", "Anything between you and the server could change what you download.",
               "Point REQUESTS_CA_BUNDLE at your organization's CA file instead, and unset HF_HUB_DISABLE_SSL_VERIFICATION."),
        "zh": ("关闭了 Hugging Face 的证书校验", "你和服务器之间的任何人都能篡改你下载的内容。", "改为把 REQUESTS_CA_BUNDLE 指向你们单位的 CA 证书，并取消 HF_HUB_DISABLE_SSL_VERIFICATION。")},
    "hf-offline": {
        "en": ("HF_HUB_OFFLINE is on", "Hugging Face libraries will use only models already downloaded.", ""),
        "zh": ("开启了 HF_HUB_OFFLINE", "Hugging Face 的库只会使用已下载的模型。", "")},
    # ---- reachability
    "hf-blocked-mirror-ok": {
        "en": ("huggingface.co cannot be reached ({kind}), but the mirror hf-mirror.com can", "{error}",
               "Set HF_ENDPOINT=https://hf-mirror.com (a third-party mirror: check your organization's policy first)."),
        "zh": ("连不上 huggingface.co（{kind}），但能连上镜像 hf-mirror.com", "{error}", "设置 HF_ENDPOINT=https://hf-mirror.com（这是第三方镜像，先确认是否符合你们单位的规定）。")},
    "net-dns": {
        "en": ("{host} cannot be found (DNS): no {purpose}", "{error}", "Check the network or the DNS server; behind a proxy, set HTTPS_PROXY."),
        "zh": ("找不到 {host}（DNS）：无法获取{purpose}", "{error}", "检查网络或 DNS；如果要走代理，设置 HTTPS_PROXY。")},
    "net-timeout": {
        "en": ("{host} does not answer (timeout): no {purpose}", "{error}", "It may be blocked; set HTTPS_PROXY, or use a mirror."),
        "zh": ("{host} 没有响应（超时）：无法获取{purpose}", "{error}", "可能被屏蔽了；设置 HTTPS_PROXY，或者使用镜像。")},
    "net-refused": {
        "en": ("{host} refused the connection: no {purpose}", "{error}", "If a proxy is set, check that it is running."),
        "zh": ("{host} 拒绝连接：无法获取{purpose}", "{error}", "如果设置了代理，确认代理正在运行。")},
    "net-ssl": {
        "en": ("{host}: the certificate check failed", "{error}. Usually a company network that inspects HTTPS.",
               "Point REQUESTS_CA_BUNDLE and SSL_CERT_FILE (and pip's `cert` setting) at your organization's CA file."),
        "zh": ("{host}：证书校验失败", "{error}。通常是单位网络在检查 HTTPS 流量。", "把 REQUESTS_CA_BUNDLE 和 SSL_CERT_FILE（以及 pip 的 cert 设置）指向你们单位的 CA 证书。")},
    "net-reset": {
        "en": ("{host}: the connection was cut off during the secure handshake: no {purpose}",
               "{error}. This is how a blocked host usually looks; it is not a certificate problem.", "Set HTTPS_PROXY, or use a mirror."),
        "zh": ("{host}：安全握手时连接被切断：无法获取{purpose}", "{error}。这通常意味着这个地址被屏蔽了，不是证书问题。", "设置 HTTPS_PROXY，或者使用镜像。")},
    "net-proxy": {
        "en": ("{host}: the proxy refused the request", "{error}", "Check the proxy address and, if it needs one, the user name and password."),
        "zh": ("{host}：代理拒绝了请求", "{error}", "检查代理地址；如果需要账号，检查用户名和密码。")},
    "net-http": {
        "en": ("{host} answered with an error", "{error}", "It may be down for a while; try again later."),
        "zh": ("{host} 返回了错误", "{error}", "可能暂时出了问题；稍后再试。")},
    "net-other": {
        "en": ("{host} cannot be reached: no {purpose}", "{error}", ""),
        "zh": ("连不上 {host}：无法获取{purpose}", "{error}", "")},
    "net-ok": {
        "en": ("The network reaches {hosts}", "", ""),
        "zh": ("网络能连上 {hosts}", "", "")},
    "net-slow": {
        "en": ("{host} answers slowly ({seconds} s)", "Large downloads from it will take a long time.", ""),
        "zh": ("{host} 响应很慢（{seconds} 秒）", "从这里下载大文件会很久。", "")},
    "ollama-running": {
        "en": ("Ollama is running ({url}, version {version})", "", ""),
        "zh": ("Ollama 正在运行（{url}，版本 {version}）", "", "")},
    "disk-low": {
        "en": ("Little free space where {cache} keeps models: {free}", "A single model is often 5–20 GB.", "Free some space, or move the cache "
               "(HF_HOME for Hugging Face, OLLAMA_MODELS for Ollama)."),
        "zh": ("{cache} 存放模型的磁盘空间不足：只剩 {free}", "一个模型通常就有 5–20 GB。", "清理一些空间，或者把缓存移到别处（Hugging Face 用 HF_HOME，Ollama 用 OLLAMA_MODELS）。")},
}


WHY = {
    "en": {"venv": "the project's environment ({origin})", "virtualenv": "the active virtualenv ({origin})",
           "conda": "the active conda environment ({origin})", "path": "the `{origin}` on PATH", "given": "{origin}"},
    "zh": {"venv": "项目的环境（{origin}）", "virtualenv": "当前激活的虚拟环境（{origin}）", "conda": "当前激活的 conda 环境（{origin}）",
           "path": "PATH 上的 `{origin}`", "given": "{origin}"},
}


class _Safe(string.Formatter):
    def get_value(self, key, args, kwargs):
        return kwargs.get(key, "?") if isinstance(key, str) else super().get_value(key, args, kwargs)


_F = _Safe()


def msg(lang: str, id_: str, **facts) -> tuple[str, str, str]:
    entry = M.get(id_)
    if entry is None:
        return id_, "", ""
    title, detail, fix = entry.get(lang) or entry["en"]
    return tuple(_F.format(s, **facts) for s in (title, detail, fix))  # type: ignore[return-value]
