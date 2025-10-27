# CANDI: Hybrid Discrete-Continuous Diffusion Models

<div align="center">
  <strong><a href="https://patrickpynadath1.github.io/candi-lander/#">🌐 Project Page</a></strong>
</div>

## Overview

CANDI (Continuous and Discrete Diffusion) is a novel hybrid diffusion model that combines continuous and discrete noise processes for high-quality text generation. This approach bridges the gap between continuous diffusion models used in computer vision and discrete token-based language modeling, offering the best of both worlds.

![CANDI Visualization](gen_imgs/candi-vis.png)

## Method

CANDI uses a hybrid kernel to coordinate both discrete and continuous corruption explicitly throughout training:

![Token Identifiability](gen_imgs/token_identifiability_fig.png)

This code base is built on the DUO codebase, which is available at this link: https://github.com/s-sahoo/duo?tab=readme-ov-file.


## Installation

1. Clone the repository:
```bash
git clone https://github.com/patrickpynadath1/candi.git
cd candi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install Flash Attention for faster training:
```bash
pip install flash-attn --no-build-isolation
```

4. Download the OWT data
```bash
bash manual_download.sh
```

## Experiments 

This codebase includes the code for running experiments on Text8 and OWT. We will integrate the QM9 experiments later. In general, we re-use the same experimental methodology and codebase from https://github.com/kuleshov-group/discrete-diffusion-guidance.

### Training

We include scripts for training models in scripts/slurm_scripts. 


### Frontier Analysis

Run temperature sweeps for frontier analysis using the following scripts:
```bash
# OpenWebText sweeps
bash scripts/gen_ppl_owt_candi_sweep.sh

# Text8 sweeps  
bash scripts/gen_text8_candi_sweep.sh
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- **Algorithm configs**: `configs/algo/` - Different diffusion algorithms (CANDI, MDLM, SEDD, etc.)
- **Data configs**: `configs/data/` - Dataset configurations
- **Model configs**: `configs/model/` - Model architecture settings
- **Base config**: `configs/config.yaml` - Main configuration file



## Key Components

- **[`algo.py`](algo.py)**: Core algorithm implementations (CANDI, MDLM, DUO, etc.)
- **[`main.py`](main.py)**: Main training and evaluation script
- **[`dataloader.py`](dataloader.py)**: Data loading and preprocessing utilities
- **[`models/`](models/)**: Model architectures (DiT)
- **[`metrics.py`](metrics.py)**: Evaluation metrics and utilities
- **[`trainer_base.py`](trainer_base.py)**: Base trainer class with common functionality

## Citation

If you use this code in your research, please cite:

```bibtex
@article{pynadath2025candi,
  title={CANDI: Hybrid Discrete-Continuous Diffusion Models},
  author={Patrick Pynadath, Jiaxin Shi, and Ruqi Zhang},
  journal={arXiv preprint},
  year={2025}
}
```



For more details, visit our [project page](https://patrickpynadath1.github.io/candi-lander/#) or check out the [paper](#).
