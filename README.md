# SpectralDiffusionLab: 

Synthetic FTIR spectrum generation using a conditional diffusion model.

## Key Features
- **Peak-Aware Diffusion**: Weighted loss for biological markers
- **HPC-Ready**: Batch script templates for Slurm/PBS
- **Medical Validation**: spectral verification and qc evaluation
- **Multi-Class**: generates both healthy/cancerous spectra

# SpectralDiffusionLab

Latent diffusion models for **FTIR-based cancer detection**.

This repository implements a **latent-space conditional diffusion model**. The model generates realistic FTIR spectra for Healthy vs Cancer patients and studies how synthetic data affects downstream classification performance.

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
├─ venv/                       # local virtual environment 
│
├─ balance_compare.py          # compare balancing strategies
├─ ddpm_sample_generate.py     # generate new Healthy/Cancer spectra
├─ Latent_ddpm_z.py            # cache AE latent codes for train/val
├─ QC.py                       # QC plots: generated vs real spectra
├─ strategic_augmentation.py   # balance-then-augment experiment 
├─ train_ae.py                 # train Conv1D autoencoder on FTIR spectra
├─ train_ddpm_latent.py        # train latent-space conditional DDPM
└─ requirements.txt
