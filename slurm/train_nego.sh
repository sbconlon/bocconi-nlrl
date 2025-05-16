#!/bin/bash
#SBATCH --job-name="train"
#SBATCH --partition=ai
#SBATCH --gpus=1
#SBATCH --output=../slurm-out/%x.%j.out   # %x = job-name, %j = job‑ID
#SBATCH --error=../slurm-err/%x.%j.err

module purge
module load nvidia/cuda-12.4.0
module load modules/miniconda3

eval "$(conda shell.bash hook)"

conda activate nlrl

echo “Running on host $(hostname)”
echo “CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES”

python train_nego.py configs/negotiation/train.json > out.txt

module purge