#!/bin/bash
#SBATCH --job-name=diffusion_test
#SBATCH --output=diffusion_test_%j.out
#SBATCH --error=diffusion_test_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # Reduced to 1 task since we're GPU-focused
#SBATCH --gres=gpu:1             # request GPUs on Unity
#SBATCH --mem=8GB
#SBATCH --partition=gpu           # Specify GPU partition
#SBATCH --mail-type=FAIL
#SBATCH --constraint=avx512
#SBATCH --mail-user=NazimAhmed_Belabbaci@student.uml.edu

module purge

# Load modules 
module load python/3.12.3
module load cuda/11.8            # Must load CUDA before PyTorch
module load PyTorch/1.13.1-foss-2022b


source ~/diffusion_venv/bin/activate


# Run your script
python3 cdm2.py
