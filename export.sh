#!/usr/bin/env bash
#
#SBATCH --job-name=wmt19-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --mem=20G
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

source setup.sh

python main.py
