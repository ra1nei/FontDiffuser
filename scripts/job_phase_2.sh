#!/bin/bash
#SBATCH -o logs/phase1_%j.out
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

module load slurm
module load python39

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dang_env

cd ~/data/fontdiffuser/FontDiffuser
git pull origin main

bash scripts/train_phase_2.sh