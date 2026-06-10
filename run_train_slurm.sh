#!/bin/bash -l
#SBATCH --job-name=job1
#SBATCH --time=1-00:00:00
#SBATCH --error=/projects/job1.err
#SBATCH --output=/projects/job1.out
#SBATCH --partition=GPUampere
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH --ntasks=1

echo "Job ID = $SLURM_JOB_ID"
echo "Job Name = $SLURM_JOB_NAME"
echo "Node List = $SLURM_NODELIST"
echo "Total Tasks = $SLURM_NTASKS"
echo "Submit Host = $SLURM_SUBMIT_HOST"
echo "CUDA Visible Devices = $CUDA_VISIBLE_DEVICES"
echo "Username = $USER"


# Activate environment
conda activate classification

export CUDA_LAUNCH_BLOCKING=1

python Training/classify.py Configs/classify_config.ini
