#!/bin/bash
# Script to deploy the project to the SLURM cluster

set -e

# Configuration
CLUSTER_USER="team04"
CLUSTER_HOST1="129.212.178.168"
CLUSTER_HOST2="134.199.193.89"
CLUSTER_PORT="32605"
REMOTE_DIR="~/coldstart"

# Choose which host to use (default to first one)
CLUSTER_HOST="${CLUSTER_HOST1}"

echo "==================================="
echo "Deploying to SLURM Cluster"
echo "==================================="
echo "User: ${CLUSTER_USER}"
echo "Host: ${CLUSTER_HOST}:${CLUSTER_PORT}"
echo "Remote directory: ${REMOTE_DIR}"
echo ""

# Create remote directory if it doesn't exist
echo "Creating remote directory..."
ssh -p ${CLUSTER_PORT} ${CLUSTER_USER}@${CLUSTER_HOST} "mkdir -p ${REMOTE_DIR}"

# Sync project files
echo "Syncing project files..."
rsync -avz -e "ssh -p ${CLUSTER_PORT}" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='venv' \
  --exclude='data' \
  --exclude='logs' \
  --exclude='.DS_Store' \
  . ${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_DIR}/

echo ""
echo "Deployment complete!"
echo ""
echo "Next steps:"
echo "1. SSH into cluster:"
echo "   ssh ${CLUSTER_USER}@${CLUSTER_HOST} -p ${CLUSTER_PORT}"
echo ""
echo "2. Navigate to project directory:"
echo "   cd ~/coldstart"
echo ""
echo "3. Submit GPU job:"
echo "   ./submit-job.sh \"flwr run . cluster-gpu\" --gpu --name xray-fed"
echo ""
echo "4. Monitor logs:"
echo "   tail -f ~/logs/job*_xray-fed.out"
