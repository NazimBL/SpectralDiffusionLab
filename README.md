# SpectralDiffusionLab

**A Latent Diffusion Framework for Generative Augmentation of FTIR Spectroscopy**

## Abstract
This repository contains the official implementation of the paper *"A Latent Diffusion Framework for Generative Augmentation of FTIR Spectroscopy"* (submitted to IJCAI 2026, AI and Health Track). The project implements a **latent-space conditional Denoising Diffusion Probabilistic Model (DDPM)** designed to generate realistic Fourier Transform Infrared (FTIR) spectra for distinguishing between Healthy and Cancerous tissue samples. 

We provide this codebase to ensure the **reproducibility** of our results. It includes the complete training pipeline, evaluation metrics, and quality control (QC) scripts.

## Reproducibility & Installation

This project uses **[uv](https://github.com/astral-sh/uv)** for fast and reliable dependency management. Reproducibility is guaranteed when installing dependencies via `uv sync`.

### Prerequisites
- Python >= 3.9
- CUDA-capable GPU (recommended for training)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

### Setup Instructions
1. **Unzip the provided archive:**
   Extract the contents of `EndoCancerFTIR_Submission.zip` to a directory of your choice.
   ```bash
   cd SpectralDiffusionLab
   ```

2. **Sync the environment:**
   Use `uv` to create the virtual environment and install exact dependencies from `uv.lock`.
   ```bash
   uv sync
   ```

3. **Activate the environment (Optional):**
   ```bash
   # On Windows
   .venv\Scripts\activate
   # On Linux/macOS
   source .venv/bin/activate
   ```
   *Alternatively, you can run scripts directly using `uv run script.py`.*

## Reproducing Paper Results

The following scripts reproduce the quantitative results (tables) and visualizations (figures) presented in the manuscript.

### Experiment I: Comparative Balancing Strategies
This experiment evaluates five balancing approaches (Original, Undersampling, SMOTE, cGAN, and LDM) as described in **Section 4.2** of the paper.
- **Reproduces:** Table 1 and Figure 4.
- **Script:** `experiment_balancing.py`

```bash
uv run experiment_balancing.py
```
*Output: Results will be saved to `Balancing_Comparison_Final_All/`.*

### Experiment II: Strategic Augmentation
This experiment investigates the scaling effects of synthetic data augmentation (0.4x to 2.0x ratios) using SMOTE, cGAN, and LDM, as described in **Section 4.3**.
- **Reproduces:** Table 2 and Figure 5.
- **Script:** `augmentation-benchmark.py`

```bash
uv run augmentation-benchmark.py
```
*Output: Results will be saved to `Strategic_Augmentation_Comparison/`.*

## Data Availability
The raw dataset used in this study (`Endo Cancer ATIR FTIR.txt`) is located in the `data/` directory. 
- **Raw Data**: `data/Endo Cancer ATIR FTIR.txt`
- **Privacy**: Patient identifiers have been anonymized. 

To reproduce the data splits used in the paper:
```bash
uv run data/data_preparation.py
```
This will generate the training and testing CSV files in `MyDataset/`.

## Training Pipeline

To retrain the models from scratch (Autoencoder, Latent DDPM) or generate new synthetic datasets, follow the pipeline sequentially:

### 1. Train the Autoencoder
Train the 1D Convolutional Autoencoder to compress spectra into a latent space.
```bash
uv run train_ae.py
```
*Outputs: Weights saved to `ldm_out/`*

### 2. Precompute Latents
Encode the entire dataset into latent vectors `z` for efficient DDPM training.
```bash
uv run Latent_ddpm_z.py
```

### 3. Train the Latent DDPM
Train the conditional diffusion model on the precomputed latents.
```bash
uv run train_ddpm_latent.py
```

### 4. Generate Synthetic Spectra
Generate new samples for both 'Healthy' and 'Cancer' classes.
```bash
uv run ddpm_sample_generate.py
```

### 5. Quality Control & Visualization
Evaluate the quality of generated spectra against real data.
```bash
uv run QC.py
```

## Repository Structure

```text
├── pyproject.toml             # Project metadata and top-level dependencies
├── uv.lock                    # Exact dependency lockfile for reproducibility
├── experiment_balancing.py    # Experiment I: Comparative Balancing Strategies
├── augmentation-benchmark.py  # Experiment II: Strategic Augmentation
├── train_ae.py                # script: Train Autoencoder
├── train_ddpm_latent.py       # script: Train Latent DDPM
├── Latent_ddpm_z.py           # script: Precompute latent representations
├── ddpm_sample_generate.py    # script: Generate synthetic samples
├── QC.py                      # script: Quality Control & Evaluation
├── baseline results/          # Results from baseline classifiers
├── data/                      # Data ingestion and preparation scripts
│   ├── Endo Cancer ATIR FTIR.txt  # Raw dataset
│   └── data_preparation.py    # Dataset splitting script
├── ldm_out/                   # Checkpoints (AE, DDPM) and logs
├── MyDataset/                 # Processed Train/Test CSVs
├── Strategic_Augmentation_Comparison/ # Output of Experiment II
├── Balancing_Comparison_Final_All/    # Output of Experiment I
└── latent space diffusion v0/ # archival/experimental versions
```

## Hardware & Environment
- **OS**: Windows / Linux
- **Python**: 3.9+ 
- **Dependencies**: See `pyproject.toml` for main libraries (PyTorch, Scikit-learn, Pandas, etc.).


