#!/bin/bash
#SBATCH --job-name=xray-fed
#SBATCH --output=/home/team04/logs/job_%j_xray-fed.out
#SBATCH --error=/home/team04/logs/job_%j_xray-fed.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

# Print job info
echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "========================================="
echo ""

# Load environment
source ~/hackathon-venv-cpu/bin/activate

# Set data directory
export DATA_DIR=/data/chest_xray_224

# Navigate to project directory
cd ~/coldstart

# Run the command
echo "Executing: flwr run"
echo ""
flwr run

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
