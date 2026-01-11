# llm-training

## Installation on Win 11

On **Windows**, using **`uv` + `pyproject.toml`**, it is currently **not possible to install CUDA-enabled PyTorch** (`+cuXXX` wheels) via dependency resolution.

This is **not a GPU, driver, or CUDA problem**.  
It is a **packaging / resolver limitation** involving:

- PyTorch CUDA wheels (`+cu121`)
- Python packaging rules (PEP 440)
- `uv`’s dependency resolver

### Symptoms

If you are affected, you will see:

```python
import torch
torch.cuda.is_available()  # False
torch.version.cuda         # None
````

And often:

```text
AssertionError: Torch not compiled with CUDA enabled
```

Even though:

* You have an NVIDIA GPU
* `nvidia-smi` works
* CUDA drivers are installed
* Python version is supported (3.11)

---


### Root Cause (Important)

#### 1. PyTorch CUDA wheels use *local version identifiers*

CUDA builds are published as:

```
torch==2.2.2+cu121
```

The `+cu121` part is a **PEP 440 local version identifier**.

Local versions:

* are valid for installed packages
* are **not first-class dependency selectors**
* cannot be reliably chosen by resolvers

---

#### 2. `uv` cannot resolve CUDA variants from `pyproject.toml`

Even if you configure:

```toml
[tool.uv.pip]
index-url = "https://download.pytorch.org/whl/cu121"
```

`uv` will still install:

```
torch==2.2.2   # CPU-only
```

This is a **known limitation** of `uv` on Windows.

> `pip install torch==2.2.2+cu121` works
> `uv sync` does **not**

This behavior is expected with current tooling.

---

#### 3. This only affects Windows users

* Linux users often use system CUDA or conda
* Windows requires bundled CUDA wheels
* CUDA selection cannot be expressed in `pyproject.toml`

---


### Solution using native Windows

#### Strategy

* Use **`uv` for environment & non-torch dependencies**
* Use **`pip` for CUDA-enabled PyTorch only**

This is intentional and widely used in ML projects.

---

### Step-by-Step Fix (Windows)

#### 1. Keep `pyproject.toml` **without PyTorch**

```toml
[project]
name = "llm-training"
version = "0.1.0"
requires-python = ">=3.11,<3.12"

dependencies = [
    "torchinfo==1.8.0",
    "transformers>=4.44.2",
    "tiktoken==0.12.0",
    "numpy<2",
    "matplotlib>=3.8",
    "requests>=2.32.5",
    "pip>=25.0",
]
```

---

#### 2. Create environment with `uv`

```powershell
uv sync
```

---

#### 3. Install CUDA PyTorch using `pip`

Before running this script, activate your virtual env

```powershell
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124
pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 --no-deps
```

---

#### 4. Verify CUDA

```python
import torch

print(torch.__version__)        # 2.2.2+cu121
print(torch.version.cuda)       # 12.1
print(torch.cuda.is_available())  # True
```

---

### Recommended Safety Check in Code

Add this early in training scripts:

```python
import torch

assert torch.cuda.is_available(), (
    "CUDA not available. "
    "Ensure PyTorch was installed with +cu121 via pip."
)
```

This prevents accidental CPU-only training.

---

### Key Takeaways

* CUDA selection **cannot be expressed** in `pyproject.toml`
* `+cu121` is **not a real dependency version**
* `uv` behavior is correct but limiting
* Mixing `uv` + `pip` is the **correct solution**
* This is a **tooling ecosystem limitation**, not user error


### Solution using WSL

1. Install Ubuntu in your WSL

```powershell
wsl --install -d Ubuntu-22.04
```

2. Set Ubuntu as default distribution

```powershell
wsl --set-default Ubuntu-22.04
```

3. Inside Ubuntu 22.04: setup Python + uv

```shell
sudo apt update
sudo apt install -y \
  python3 \
  python3-venv \
  python3-dev \
  build-essential \
  git \
  curl \
  ca-certificates


curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

You can verify if it installed
```shell
uv --version
python3 --version
```


4. Verify if Nvidia is visible from WSL

```shell
nvidia-smi
```

5. Create new venv

```shell
uv venv .venv
```

6. Activate venv and install dependencies

```shell
source .venv/bin/activate
uv sync
```

7. Test it

```shell
export PYTHONPATH="$PYTHONPATH:/mnt/c/REPOSITORIES/llm-training/src"
uv run python src/llms/build_llm_transfomer_model_multi_head_attantion.py
```

### Link to reported issues
- [Reported issue in StackOverflow](https://stackoverflow.com/questions/79829472/pytorch-installed-via-uv-project-shows-cpu-only-version-on-windows-with-cuda-spe?utm_source=chatgpt.com)
- [Reported issue in GitHub - 7202 - Issues creating a cuda-enabled pytorch environment with UV](https://github.com/astral-sh/uv/issues/7202)
- [Reported issue in GitHub - 1855 - Local version identifiers are not ignored when testing version equality](https://github.com/astral-sh/uv/issues/1855)