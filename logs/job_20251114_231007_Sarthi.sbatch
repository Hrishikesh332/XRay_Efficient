#!/bin/bash
#SBATCH --job-name=Sarthi
#SBATCH --output=/home/team04/logs/job_%j_Sarthi.out
#SBATCH --error=/home/team04/logs/job_%j_Sarthi.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Print job info
echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================="
echo ""

# Load environment
source ~/hackathon-venv/bin/activate

# Set data directory
export DATA_DIR=/data/chest_xray_224

# Navigate to project directory
cd ~/coldstart

# Run the command
echo "Executing: flwr run . cluster-gpu"
echo ""
flwr run . cluster-gpu

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
