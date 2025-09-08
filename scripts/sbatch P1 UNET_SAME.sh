#!/bin/bash
#SBATCH -o logs/phase1_%j.out
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

cd "$(dirname "$0")"

module load slurm
module load python39

source ~/miniconda3/bin/activate dang_env

rm -f ../../outputs/FontDiffuser/fontdiffuser_training.log

cd ~/data/fontdiffuser/FontDiffuser
git pull origin main

bash "scripts/P1 UNET_SAME.sh"