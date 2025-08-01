#!/bin/bash
#SBATCH -o logs/phase1_%j.out              # Ghi log output (%j = JobID)
#SBATCH --time=12:00:00                    # Thời gian tối đa (12 tiếng, chỉnh lại tùy bạn)
#SBATCH --gres=gpu:1                       # Dùng 1 GPU
#SBATCH -N 1                               # Số lượng node
#SBATCH --ntasks=1                         # 1 task
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4                  # 4 CPU core cho mỗi task

# ===== Load module Slurm và Python =====
module load slurm
module load python39

# ===== (Tùy chọn) Activate môi trường ảo nếu cần =====
# Nếu bạn dùng virtualenv:
source ~/dang_env/bin/activate

# ===== Chạy script train =====
bash scripts/train_phase_1.sh