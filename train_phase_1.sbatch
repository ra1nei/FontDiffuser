#!/bin/bash
#SBATCH -J fontdiff_phase1               # Tên job
#SBATCH -o logs/train_phase_1.out        # File log output
#SBATCH -e logs/train_phase_1.err        # File log error
#SBATCH --time=02:00:00                  # Thời gian tối đa (2h ở đây)
#SBATCH --gres=gpu:1                     # Sử dụng 1 GPU
#SBATCH -N 1                             # 1 node
#SBATCH --ntasks=1                       # 1 task
#SBATCH --ntasks-per-node=1              # 1 task per node
#SBATCH --cpus-per-task=4                # 4 CPU core cho mỗi task

# === Load môi trường ===
module load python39                     # Load python 3.9, bạn có thể thay nếu dùng môi trường khác

# === (Tuỳ chọn) Kích hoạt env nếu dùng conda ===
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate dang_env

# === Chạy script huấn luyện ===
bash train_phase_1.sh