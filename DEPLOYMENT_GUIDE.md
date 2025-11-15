# Deployment Guide for SLURM Cluster

This guide provides step-by-step instructions for deploying and running the federated chest x-ray classification project on the hackathon SLURM cluster.

## Prerequisites

- SSH access to the cluster (team04 credentials)
- Git installed locally (for version control)
- Basic familiarity with command line

## Quick Start

### Option 1: Using the Deploy Script (Recommended)

```bash
# From your local machine, in the coldstart directory
./deploy.sh
```

This will automatically sync all files to the cluster.

### Option 2: Manual Deployment

```bash
# Sync files manually using rsync
rsync -avz -e "ssh -p 32605" coldstart/ team04@129.212.178.168:~/coldstart/

# Or using scp
scp -P 32605 -r coldstart/ team04@129.212.178.168:~/coldstart/
```

### Option 3: Using Git (Recommended for Tracking Changes)

```bash
# Initialize git repository locally (if not already done)
cd coldstart
git init
git add .
git commit -m "Initial federated learning implementation"

# Add remote repository
git remote add origin https://github.com/SarthiBorkar/SarthiBorkar.github.io.git
git push -u origin main

# On the cluster, clone the repository
ssh team04@129.212.178.168 -p 32605
cd ~
git clone https://github.com/SarthiBorkar/SarthiBorkar.github.io.git coldstart
```

## Running on the Cluster

### 1. Connect to Cluster

```bash
ssh team04@129.212.178.168 -p 32605
# Password: Et;;%oiWS)IT<Ot0uY0CTrG07YiHZbSQ

# Alternative host:
# ssh team04@134.199.193.89 -p 32605
```

### 2. Navigate to Project Directory

```bash
cd ~/coldstart
```

### 3. Submit Training Job

#### For GPU Training (Recommended for Best Performance)

```bash
./submit-job.sh "flwr run . cluster-gpu" --gpu --name xray-fed-gpu
```

#### For CPU Training (For Testing)

```bash
./submit-job.sh "flwr run . cluster-cpu" --name xray-fed-cpu
```

### 4. Monitor Job Progress

```bash
# View logs in real-time
tail -f ~/logs/job*_xray-fed*.out

# Check job status
squeue -u team04

# View completed job logs
ls -lth ~/logs/
cat ~/logs/job_XXX_xray-fed-gpu.out
```

## Configuration Options

### Adjusting Training Parameters

Edit `pyproject.toml` to change:

```toml
[tool.flwr.app.config]
num-server-rounds = 50          # Number of federated rounds
fraction-fit = 1.0              # Fraction of clients for training
fraction-evaluate = 1.0         # Fraction of clients for evaluation

[tool.flwr.federations.local-simulation]
options.num-supernodes = 3      # Number of clients (3 hospitals)
```

### Client Configuration

Modify training hyperparameters by editing run config values in the code:

- `batch-size`: Batch size for training (default: 32)
- `image-size`: Input image size (default: 224)
- `local-epochs`: Epochs per round (default: 1)
- `learning-rate`: Initial learning rate (default: 1e-4)

## Data Directory Setup

The project expects data in the following structure:

```
/data/chest_xray_224/
├── hospital_A/
│   ├── train/
│   │   ├── 0/  (No Finding)
│   │   └── 1/  (Any Finding)
│   └── eval/
│       ├── 0/
│       └── 1/
├── hospital_B/
│   └── ...
└── hospital_C/
    └── ...
```

The cluster should have preprocessed datasets at:
- `/data/chest_xray_128/` (128x128 images)
- `/data/chest_xray_224/` (224x224 images - default)

## Troubleshooting

### Job Not Starting

```bash
# Check job queue
squeue -u team04

# Check if resources are available
sinfo
```

### Out of Memory Errors

Reduce batch size in the code or use smaller image size (128 instead of 224).

### Data Not Found

Check that the `DATA_DIR` environment variable points to correct location:

```bash
echo $DATA_DIR
# Should show: /data/chest_xray_224
```

### Import Errors

The cluster should have all dependencies pre-installed in `hackathon-venv` (GPU) or `hackathon-venv-cpu` (CPU). If you encounter import errors, contact support.

## Experiment Tracking

### Viewing Results

Training metrics are printed to the log files in `~/logs/`. Key metrics include:

- `train_loss`: Training loss per client
- `train_accuracy`: Training accuracy per client
- `auroc`: **Primary metric** - Area Under ROC Curve
- `accuracy`: Evaluation accuracy
- `precision`, `recall`, `f1`: Additional classification metrics

### Saving Best Model

The best performing model (based on AUROC) should be saved. You can modify `server_app.py` to add checkpointing:

```python
# Add to FedAvgWithScheduling class
def aggregate_evaluate(self, server_round, results, failures):
    aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
        server_round, results, failures
    )

    # Save checkpoint if best AUROC
    if aggregated_metrics and 'auroc' in aggregated_metrics:
        auroc = aggregated_metrics['auroc']
        if auroc > self.best_auroc:
            self.best_auroc = auroc
            # Save model checkpoint
            log(INFO, f"New best AUROC: {auroc:.4f}")

    return aggregated_loss, aggregated_metrics
```

## Advanced: Hyperparameter Tuning

To experiment with different hyperparameters, create multiple job submissions:

```bash
# Experiment 1: Default settings
./submit-job.sh "flwr run . cluster-gpu" --gpu --name exp1-default

# Experiment 2: Higher learning rate
# (modify pyproject.toml or code first)
./submit-job.sh "flwr run . cluster-gpu" --gpu --name exp2-high-lr

# Experiment 3: More local epochs
./submit-job.sh "flwr run . cluster-gpu" --gpu --name exp3-more-epochs
```

## Performance Tips

1. **Use GPU**: Always use `--gpu` flag for significantly faster training
2. **Batch Size**: Increase if you have memory (try 64 or 128)
3. **Image Size**: Use 224x224 for better accuracy (EfficientNet is optimized for this)
4. **Learning Rate**: Start with 1e-4, decay over rounds
5. **Local Epochs**: Keep at 1-2 per round to avoid overfitting to local data

## Getting Help

If you encounter issues:
1. Check the logs in `~/logs/`
2. Verify data directory structure
3. Contact support in #support channel
4. Check Flower documentation: https://flower.ai/docs/

## Next Steps

After successful training:
1. Analyze results from logs
2. Compare AUROC scores across rounds
3. Experiment with different architectures or strategies
4. Implement model checkpointing and evaluation on test set
