#!/bin/bash
#SBATCH --job-name=hospital_A
#SBATCH --output=/home/team04/logs/job_%j_hospital_A.out
#SBATCH --error=/home/team04/logs/job_%j_hospital_A.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
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
export DATA_DIR=${HOME}/xray-data/xray_fl_datasets_preprocessed_224

# Navigate to project directory
cd ~/coldstart

# Create models directory
mkdir -p models

# Run training
echo "Training Hospital A..."
echo "Data directory: ${DATA_DIR}"
echo ""

python train_single_hospital.py \
    --hospital A \
    --data-dir ${DATA_DIR} \
    --epochs 5 \
    --batch-size 32 \
    --lr 0.0001 \
    --image-size 224 \
    --output-dir models

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
