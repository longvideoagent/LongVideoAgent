# Installation

### Preferred Method

First, create a virtual environment named `lvagent`, clone our repository, and perform the installation in the project directory.

```bash
conda create -n lvagent python=3.11
conda activate lvagent
cd ./VideoQAgent
# Install verl
pip install -e .
# Install flash-attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

If you encounter any unresolved conflicts during installation, try the following alternative approach:

```bash
# Install verl without dependencies
pip install -e . --no-deps
# Install other dependencies
pip install -r requirements.txt
```

### Optional Environment Combination

We provide a runnable environment setup as an optional reference: `cuda12.4`, `vllm==0.7.3`, `torch==2.5.1 cu124`, and `transformers==4.57.6`.
