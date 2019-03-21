#!/usr/bin/env bash
#
#SBATCH --job-name=wmt19-train
#SBATCH --nodes=1
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=log.out
#SBATCH --error=log.err

conda activate ruse-remake
module load cuda91/toolkit/9.1.85
module load cudnn/7.0-cuda_9.1

python main.py
