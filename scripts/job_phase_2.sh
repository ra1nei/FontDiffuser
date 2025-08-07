#!/bin/bash
#SBATCH -o logs/phase1_%j.out
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

module load slurm
module load python39

source ~/dang_env/bin/activate

bash scripts/train_phase_2.sh