#!/bin/bash
#SBATCH --job-name=ftir_ablate                 # job name
#SBATCH --output=logs/ablation_%a.out          # stdout per array index
#SBATCH --error=logs/ablation_%a.err           # stderr per array index
#SBATCH --array=1-4                            # run modes 1–4
#SBATCH --time=2-00:00:00                      # 2 days walltime
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                    # 1 task (GPU-focused)
#SBATCH --gres=gpu:1                           # request one GPU
#SBATCH --mem=30GB                             # host memory
#SBATCH --partition=gpu                        # Unity GPU partition

set -euo pipefail

# ensure folders exist
mkdir -p logs runs/ablation_v1

# --- Modules / env (tweak to your Unity env) ---
module purge
# If your cluster provides these modules, great; if not, no-op.
module load cuda/12.1 2>/dev/null || true


# Prevent OpenMP/MKL oversubscription
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

MODE="${SLURM_ARRAY_TASK_ID}"
RUN_DIR="runs/ablation_v1"

# Point these to your actual CSVs (same columns across splits)
TRAIN_CSV="train_set.csv"
VAL_CSV="val_set.csv"
TEST_CSV="test_set.csv"

echo "=== Starting ablation mode ${MODE} on $(hostname) at $(date) ==="

srun python3 cdm_ablation.py \
  --train_csv "${TRAIN_CSV}" \
  --val_csv   "${VAL_CSV}" \
  --test_csv  "${TEST_CSV}" \
  --epochs 50 \
  --batch_size 64 \
  --lr 5e-5 \
  --timesteps 250 \
  --guidance_scale 2.0 \
  --grad_clip 5.0 \
  --num_workers 4 \
  --modes "${MODE}" \
  --save_dir "${RUN_DIR}"

echo "=== Finished mode ${MODE} at $(date) ==="
