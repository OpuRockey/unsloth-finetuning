# LLM Fine-Tuning Environment Setup Guide
## Windows 11 + Python + uv + Unsloth

This guide covers dependency installation, environment verification, and GPU/CUDA checks for a code-first LLM fine-tuning project using Unsloth on Windows 11.

## 1. Prerequisites

- Windows 11
- Python 3.13.x
- `uv` installed
- NVIDIA GPU and compatible NVIDIA drivers
- PowerShell terminal

**Important:** Unsloth, PyTorch, and CUDA-related packages must be compatible with each other. Follow the [official Unsloth Windows installation guide](https://unsloth.ai/docs/get-started/install/windows-installation) for the current installation requirements.

## 2. Create the project environment

Open PowerShell and navigate to your project directory:

```powershell
cd E:\learning-fundamentals\fine-tuning\unsloth-finetuning
```

If this is a new project, initialize it:

```powershell
uv init --python 3.13.14
```

Create the virtual environment:

```powershell
uv venv --python 3.13.14
```

If the project requires the exact Python patch version 3.13.14:

```powershell
uv python install 3.13.14
uv venv --python 3.13.14 --clear
```

**Note:** Use `--clear` only when you intend to recreate the existing virtual environment. This removes its installed packages.

Activate the virtual environment:
```
.\.venv\Scripts\Activate.ps1
```


Verify the Python version and executable:

```powershell
uv run python --version
uv run python -c "import sys; print(sys.executable)"
```

The executable path should point to the project's `.venv\Scripts\python.exe`.

## 3. Install project dependencies

### 3.1 Install standard dependencies

Use `uv add` to declare dependencies in `pyproject.toml`, update `uv.lock`, and synchronize the environment.

### 3.2 Install Unsloth

Unsloth has specialized installation requirements for GPU acceleration and compatible PyTorch builds. Follow its current Windows installation guide rather than assuming a generic installation command will work for every setup.

If the official instructions specify the following command for your configuration, run it from the project directory:

```powershell
uv pip install unsloth --torch-backend=auto
```

This installs Unsloth and its required packages into the selected Python environment. Confirm that the command targets your project environment, not a global Python installation.

## 4. Verify the dependency installation

Run the following checks from the project directory.

### 4.1 Verify installed packages

```powershell
uv pip list --python .venv\Scripts\python.exe
```

Check that `datasets`, `transformers`, `trl`, `peft`, and `unsloth` appear in the package list.

You can also check individual packages:

```powershell
uv pip show datasets
uv pip show transformers
uv pip show trl
uv pip show peft
uv pip show unsloth
```

If a package is missing, check whether it was installed into a different Python environment.

### 4.2 Verify imports

```powershell
uv run python -c "import datasets, transformers, trl, peft; print('Standard dependencies imported successfully')"
```

Then verify Unsloth separately:

```powershell
uv run python -c "import unsloth; print('Unsloth imported successfully'); print(unsloth.__file__)"
```

The reported Unsloth path should belong to the project's `.venv`.

If an import fails, read the complete error message before reinstalling packages. The cause may be a missing dependency, an incompatible version, or a PyTorch/CUDA issue.

### 4.3 Print package versions

```powershell
uv run python -c "import datasets, transformers, trl, peft, torch; print('datasets:', datasets.__version__); print('transformers:', transformers.__version__); print('trl:', trl.__version__); print('peft:', peft.__version__); print('torch:', torch.__version__)"
```

This helps diagnose compatibility problems and document the environment.

To verify the project dependency declarations:

```powershell
uv tree
```

Review `pyproject.toml` to confirm that the dependencies you intend to manage are listed under `[project]`.

## 5. Verify NVIDIA GPU and CUDA availability

Successful package installation does not guarantee that GPU acceleration works.

### 5.1 Check the NVIDIA driver

```powershell
nvidia-smi
```

Confirm that your NVIDIA GPU appears in the output.

If the command fails, check your NVIDIA driver installation first.

### 5.2 Check PyTorch and CUDA

```powershell
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA build:', torch.version.cuda); print('GPU count:', torch.cuda.device_count())"
```

Expected results for a working GPU setup:

- `CUDA available: True`
- `GPU count` is at least `1`
- `PyTorch CUDA build` reports a CUDA version rather than `None`

Check the detected GPU name:

```powershell
uv run python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA GPU unavailable')"
```

The GPU name should match your NVIDIA graphics card.

**Important:** The CUDA version reported by PyTorch is the CUDA runtime version associated with its build. It does not need to match the version displayed by `nvcc`, and a separate CUDA Toolkit installation is not always required. The NVIDIA driver must be compatible with the installed PyTorch CUDA runtime.

## 6. Verify Unsloth can access the GPU

Once the package imports and PyTorch reports CUDA availability, perform a basic Unsloth import check:

```powershell
uv run python -c "from unsloth import FastLanguageModel; print('Unsloth model loader imported successfully')"
```

This verifies that the model-loading interface can be imported. It does not prove that a model can train successfully.

The next practical test is to load a small supported model using the official Unsloth workflow, then verify that the model loads without CUDA or compatibility errors.

## 7. Understand `uv add` versus `uv pip install`

| Command | Updates `pyproject.toml` | Updates `uv.lock` | Installs packages |
|---|---|---|---|
| `uv add package` | Yes | Yes | Yes, synchronizes the environment |
| `uv pip install package` | No | No | Yes, in the selected environment |
| `uv add package --no-sync` | Yes | Yes | Does not synchronize the environment |

Running both `uv pip install datasets transformers trl peft` and `uv add datasets transformers trl peft` is generally harmless, but redundant. The second command manages the project dependencies and synchronizes the environment.

For ordinary dependencies, prefer `uv add`. For Unsloth and GPU-sensitive dependencies, follow its official installation instructions and verify the resulting environment.

## 8. Troubleshooting

### Package installed but import fails

Check which Python is running:

```powershell
uv run python -c "import sys; print(sys.executable)"
```

Check the package in the project environment:

```powershell
uv pip show --python .venv\Scripts\python.exe transformers
```

### CUDA is unavailable

1. Run `nvidia-smi`.
2. Check `torch.__version__` and `torch.version.cuda`.
3. Verify that the installed PyTorch build supports CUDA.
4. Review the Unsloth installation instructions for your Windows configuration.
5. Avoid installing arbitrary PyTorch versions without checking compatibility.

### `uv` reports hardlink warnings

A message such as `Failed to hardlink files; falling back to full copy` is usually a performance warning, not an installation failure. Verify the installation before taking further action.

### Dependencies change unexpectedly

Before running `uv sync`, review the dependency declarations and lockfile. Synchronization may change installed packages to match the lockfile, potentially affecting GPU compatibility if the environment was configured separately.

## 9. Final verification checklist

- [ ] Project uses the intended Python version.
- [ ] `sys.executable` points to `.venv`.
- [ ] `datasets` is installed and imports successfully.
- [ ] `transformers` is installed and imports successfully.
- [ ] `trl` is installed and imports successfully.
- [ ] `peft` is installed and imports successfully.
- [ ] `unsloth` is installed and imports successfully.
- [ ] `pyproject.toml` contains the intended dependencies.
- [ ] `uv.lock` is consistent with the intended dependency workflow.
- [ ] `nvidia-smi` detects the NVIDIA GPU.
- [ ] `torch.cuda.is_available()` returns `True`.
- [ ] PyTorch detects the correct GPU.
- [ ] A small supported Unsloth model loads successfully before beginning a training run.

**Recommended workflow:** Set up the virtual environment, install ordinary dependencies with `uv add`, install Unsloth according to its official Windows instructions, and verify package imports and GPU availability before training.

### 10 Useful commands
```
uv tree
uv add --help
uv pip install --help
uv pip list
uv pip freeze > requirements-windows.txt
uv pip check
uv pip freeze > requirements-windows.txt
```

### 11 Ruff linting
```
uv add --dev ruff // installs ruff as uv dependency
uv run ruff check src
uv run ruff check src/ --fix
uv run ruff check --select I --fix src/
```