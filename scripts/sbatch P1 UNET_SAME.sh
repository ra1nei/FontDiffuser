#!/bin/bash
#SBATCH -o logs/phase1_%j.out
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# Vào project
cd ~/data/fontdiffuser/FontDiffuser || exit

# Load conda đúng path
source /data/cndt_hangdv/miniconda3/bin/activate fontdiffuser || { echo "Failed to activate conda env"; exit 1; }

# Xóa log cũ
rm -f outputs/FontDiffuser/fontdiffuser_training.log

# Cập nhật code
git pull origin main

# Chạy training
bash "scripts/P1 UNET_SAME.sh"
