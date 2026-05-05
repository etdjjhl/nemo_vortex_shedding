# Environment Setup Notes

This file documents the actual environment used for vortex_shedding_mgn training,
including all manual fixes made during setup. Use this as the reference when
writing or updating the environment setup script.

---

## Base Environment

| Item | Value |
|------|-------|
| Conda env path | `/home/chenguanfeng/work/env_conda/nemo` |
| Python | 3.11.15 |
| OS | Linux (WSL2, kernel 6.6.87.2-microsoft-standard-WSL2) |
| GPU | NVIDIA GeForce RTX 3090 (24 GB) |
| CUDA | 12.1 |

---

## Core Packages

| Package | Version | Install Command | Notes |
|---------|---------|----------------|-------|
| `torch` | 2.5.1+cu121 | `pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121` | **必须 ≥ 2.5**，见下方说明 |
| `torch_geometric` | 2.7.0 | `pip install torch_geometric` | 标准安装即可 |
| `torch_scatter` | 2.1.2+pt24cu121 | `pip install torch_scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html` | **必须从 PyG wheel 源安装**，版本与 torch 绑定 |
| `physicsnemo` | 2.0.0 | 已预装于 conda env | 见下方 Patch 说明 |
| `tfrecord` | 1.14.6 | `pip install tfrecord` | 读取 DeepMind TFRecord 数据集 |
| `wandb` | 0.26.1 | `pip install wandb` | 代码中有引用，需安装但默认 disabled |
| `hydra-core` | 1.3.2 | 已预装 | |
| `tensorboard` | 2.20.0 | 已预装 | 用于训练曲线可视化 |
| `nvidia-ml-py` | 13.595.45 | `pip install nvidia-ml-py` | GPU 监控，替代已弃用的 `pynvml` |
| `scipy` | 1.17.1 | 已预装 | |

---

## 关键问题与修复

### 问题 1：physicsnemo 安装的 PyTorch 版本不兼容

**现象：** 运行 `train.py` 报错
```
AttributeError: 'LayerNorm' object has no attribute 'register_load_state_dict_pre_hook'
```

**原因：** conda env 预装的是 PyTorch 2.4.0，但 physicsnemo 2.0.0 要求 PyTorch ≥ 2.5（该方法在 2.5 中才加入）。

**修复：** 升级 PyTorch 到 2.5.1，同时重装 torch_scatter（wheel 与 torch 版本绑定）。
```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

---

### 问题 2：physicsnemo 启动时 DiT 模型导入失败

**现象：** `import physicsnemo.models.meshgraphnet` 时报错
```
ModuleNotFoundError: No module named 'torch.distributed.tensor.placement_types'
```

**原因：** `physicsnemo/models/__init__.py` 无条件 import 所有模型（包括 DiT），而 DiT 依赖 PyTorch 内部模块，在某些版本下不可用。

**修复（已直接 patch 安装包文件）：**

文件路径：
```
/home/chenguanfeng/work/env_conda/nemo/lib/python3.11/site-packages/physicsnemo/models/__init__.py
```

将原来的：
```python
from .dit import DiT
from .domino import DoMINO
from .mlp import FullyConnected
```

改为：
```python
try:
    from .dit import DiT
except (ImportError, ModuleNotFoundError):
    pass  # DiT requires PyTorch >= 2.5; skip if unavailable

try:
    from .domino import DoMINO
except (ImportError, ModuleNotFoundError):
    pass

from .mlp import FullyConnected
```

> ⚠️ 这是对已安装包的直接修改，如果重建 conda env 需要重新 patch。
> 建议在环境脚本中用 `sed` 或 Python 脚本自动完成此 patch。

---

### 问题 3：pynvml 弃用警告

**现象：** 启动时打印
```
FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead.
```

**修复：**
```bash
pip uninstall pynvml -y
pip install nvidia-ml-py
```

`nvidia-ml-py` 的 Python import 名称仍为 `pynvml`，代码无需修改。

---

## 环境搭建完整脚本（草稿）

```bash
#!/usr/bin/env bash
# 在已有的 conda env /home/chenguanfeng/work/env_conda/nemo 基础上补装依赖

PYTHON=/home/chenguanfeng/work/env_conda/nemo/bin/python
PIP=/home/chenguanfeng/work/env_conda/nemo/bin/pip

# 1. 升级 PyTorch 到 2.5.1 (CUDA 12.1)
$PIP install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. PyG 及其依赖（wheel 版本必须与 torch 对应）
$PIP install torch_geometric
$PIP install torch_scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# 3. 其他项目依赖
$PIP install tfrecord wandb nvidia-ml-py
$PIP uninstall pynvml -y 2>/dev/null || true

# 4. Patch physicsnemo models/__init__.py（绕过 DiT 导入失败）
MODELS_INIT=$($PYTHON -c "import physicsnemo; import os; print(os.path.join(os.path.dirname(physicsnemo.__file__), 'models', '__init__.py'))")
$PYTHON - <<'EOF'
import re, sys

path = sys.argv[1]
with open(path) as f:
    src = f.read()

old = "from .dit import DiT\nfrom .domino import DoMINO\nfrom .mlp import FullyConnected"
new = """try:
    from .dit import DiT
except (ImportError, ModuleNotFoundError):
    pass  # DiT requires PyTorch >= 2.5; skip if unavailable

try:
    from .domino import DoMINO
except (ImportError, ModuleNotFoundError):
    pass

from .mlp import FullyConnected"""

if old in src:
    with open(path, "w") as f:
        f.write(src.replace(old, new))
    print(f"Patched: {path}")
else:
    print(f"Already patched or unexpected content in: {path}")
EOF
$MODELS_INIT

echo "Environment setup complete."
```

---

## 性能实验记录（2026-05-05）

训练配置优化历程（quick_test 基准：20 samples，50 time steps，2 epochs）：

| 版本 | batch_size | amp | persistent_workers | prefetch_factor | avg epoch时间 | GPU 利用率 |
|------|-----------|-----|--------------------|----------------|:---:|:---:|
| 初始（调试）| 1 | False | False | — | ~45s | ~45% |
| 优化后 | 4 | True | True | 4 | ~14.5s | ~66% |
| **当前最优** | **8** | **True** | **True** | **4** | **~10.8s** | **~77%** |

> 注：从 45s→14.5s 的大幅提升中，约一半来自 `persistent_workers + prefetch_factor`，另一半来自 PyTorch CUDA kernel 编译缓存热身完成（首次使用 AMP 时有一次性开销）。
