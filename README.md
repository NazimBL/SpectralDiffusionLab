# SpectralDiffusionLab: Spectral Generation for Cancer Detection


Generates synthetic FTIR spectra using conditioned diffusion models for endoscopic cancer research.

## Key Features
- **Peak-Aware Diffusion**: Weighted loss for biological markers
- **HPC-Ready**: Batch script templates for Slurm/PBS
- **Medical Validation**: Built-in spectral verification tools
- **Multi-Class**: Handles both healthy/cancerous spectra

# SpectralDiffusionLab

Latent diffusion models for **FTIR-based endometrial cancer detection**.

This repository implements a **latent-space conditional diffusion model** on top of a 1D convolutional autoencoder. The model generates realistic FTIR spectra for Healthy vs Cancer patients and studies how synthetic data affects downstream classification (XGBoost) performance.

---

## Repository structure

```text
├─ baseline results/           # CSVs / plots from baseline & experiments
├─ data/                       # data parsing & preparation scripts + raw/parsed files
│  ├─ data_distribution.py     # plot patient counts per class
│  ├─ data_parsing.py          # parse IRootLab export -> parsed CSV
│  ├─ data_preparation.py      # build Healthy vs Cancer train/test sets
│  ├─ Endo Cancer ATIR FTIR.txt #original dataset
│  ├─ ftir_raw_parsed.xlsx
│  └─ patient_counts_per_class.png
├─ latent space diffusion v0/  # early / experimental versions of the pipeline
├─ ldm_out/                    # autoencoder & DDPM weights, logs, generated spectra
├─ legacy/                     # older raw-space diffusion code and experiments
├─ MyDataset/                  # processed train/test CSVs used by the models
├─ venv/                       # local virtual environment (ignored in git)
│
├─ balance_compare.py          # compare balancing strategies (XGBoost)
├─ ddpm_sample_generate.py     # generate new Healthy/Cancer spectra
├─ Latent_ddpm_z.py            # cache AE latent codes for train/val
├─ QC.py                       # QC plots: generated vs real spectra
├─ strategic_augmentation.py   # balance-then-augment experiment (XGBoost)
├─ train_ae.py                 # train Conv1D autoencoder on FTIR spectra
├─ train_ddpm_latent.py        # train latent-space conditional DDPM
└─ requirements.txt
