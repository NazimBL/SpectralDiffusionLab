# SpectralDiffusionLab: Spectral Generation for Cancer Detection


Generates synthetic FTIR spectra using conditioned diffusion models for endoscopic cancer research.

## Key Features
- **Peak-Aware Diffusion**: Weighted loss for biological markers
- **HPC-Ready**: Batch script templates for Slurm/PBS
- **Medical Validation**: Built-in spectral verification tools
- **Multi-Class**: Handles both healthy/cancerous spectra

## Quick Start

# Install
pip install -r requirements.txt

# Generate samples
python generate.py --class 1 --steps 200 --guidance 1.5
