#!/bin/bash
#SBATCH -o logs/phase1_%j.out
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

# Load biến môi trường từ ~/.bashrc (WANDB_API_KEY sẽ có ở đây)
source ~/.bashrc

# Vào project
cd ~/data/fontdiffuser/FontDiffuser || exit

# Đảm bảo wandb login với đúng key của bạn
wandb login --relogin $WANDB_API_KEY

# Load conda env
source /data/cndt_hangdv/miniconda3/bin/activate fontdiffuser || { echo "Failed to activate conda env"; exit 1; }

# Xóa log cũ
rm -f outputs/FontDiffuser/fontdiffuser_training.log

# Cập nhật code
git pull origin main

# Chạy training
bash "scripts/P2 UNET_SAME_INTRA.sh"