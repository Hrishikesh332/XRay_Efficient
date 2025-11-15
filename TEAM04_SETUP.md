# Team04 - Modified Hackathon Template

## ✅ What We Changed

This is the **official hackathon template** with one key modification:
- **Model changed from ResNet18 to EfficientNet B0** (as requested)
- Uses ImageNet pretrained weights
- Adapted for grayscale X-ray images
- Binary classification with dropout for regularization

## 🚀 Quick Start on Server

### 1. Upload Template to Server

From Windows (in PowerShell):
```bash
cd "C:\Users\handa\Documents\AI revolution\Coding AI agents\distributed-ai-hack-berlin"
scp -P 32605 -r coldstart team04@129.212.178.168:~/
```

Password: `Et;;%oiWS)IT<Ot0uY0CTrG07YiHZbSQ`

### 2. SSH into Server

```bash
ssh team04@129.212.178.168 -p 32605
```

### 3. Setup Environment

```bash
cd ~/coldstart

# Activate GPU environment
source ~/hackathon-venv/bin/activate

# Install template dependencies
pip install -e .
```

This will install:
- flwr[simulation]>=1.23.0
- ray>=2.0.0
- torch, torchvision
- datasets, scipy, scikit-learn
- wandb (for logging)
- jupyterlab, marimo, pandas, matplotlib, seaborn

### 4. Configure W&B (Optional but Recommended)

Get your W&B credentials from hackathon organizers, then:

```bash
export WANDB_API_KEY="your_api_key_here"
export WANDB_PROJECT="coldstart2025-team04"
```

Or edit `cold_start_hackathon/server_app.py` lines 25-26 to hardcode them.

### 5. Submit Training Job

```bash
# Basic training (100 rounds, learning rate 0.01)
./submit-job.sh "flwr run . cluster --stream" --gpu

# With custom name for tracking
./submit-job.sh "flwr run . cluster --stream" --gpu --name efficientnet_b0_lr001

# Test with fewer rounds first
# Edit pyproject.toml to change num-server-rounds from 100 to 10
./submit-job.sh "flwr run . cluster --stream" --gpu --name test_run
```

### 6. Monitor Training

```bash
# Check job status
squeue -u team04

# View logs
tail -f ~/logs/efficientnet_b0_lr001_*.out

# Check W&B dashboard
# https://wandb.ai/coldstart2025-team04/coldstart2025
```

### 7. Find Your Best Model

After training completes:
```bash
ls -lh ~/models/

# Models are saved as: {run_name}_round{N}_auroc{XXXX}.pt
# Example: efficientnet_b0_lr001_round25_auroc7845.pt
```

### 8. Evaluate Your Model

```bash
# Update evaluate.py with your best model path
nano evaluate.py
# Change line 19: MODEL_PATH = "/home/team04/models/efficientnet_b0_lr001_round25_auroc7845.pt"

# Run evaluation
./submit-job.sh "python evaluate.py" --gpu --name eval_final
```

## ⚙️ Configuration

Edit `pyproject.toml` to change hyperparameters:

```toml
[tool.flwr.app.config]
image-size = 128  # Image size (128 or 224)
num-server-rounds = 100  # Number of federated rounds
local-epochs = 1  # Epochs per hospital per round
lr = 0.01  # Learning rate
```

**Recommendations:**
- **Start with image-size=128** (faster, fits in memory)
- **Reduce num-server-rounds to 20-30** for 20-minute time limit
- **Try lr=0.001** (lower learning rate for pretrained model)

## 📊 Key Differences from Official Template

| Aspect | Official Template | Our Version |
|--------|------------------|-------------|
| **Model** | ResNet18 (no pretrained) | **EfficientNet B0 (pretrained)** |
| **Parameters** | ~11M | ~5M |
| **Input** | Grayscale (1 channel) | Grayscale (1 channel) |
| **Output** | Binary (1 logit) | Binary (1 logit) |
| **Dropout** | No | **Yes (0.3)** |

## 📂 Important Files

### Main Code
- `cold_start_hackathon/task.py` - **Model definition (EfficientNet B0)**
- `cold_start_hackathon/client_app.py` - Client training logic
- `cold_start_hackathon/server_app.py` - Server aggregation & W&B logging
- `cold_start_hackathon/util.py` - Metrics & model saving

### Configuration
- `pyproject.toml` - **Main configuration file**
- `evaluate.py` - Final evaluation script

### Dataset Locations (on server)
- Raw: `/shared/hackathon/datasets/xray_fl_datasets/`
- **Preprocessed (128x128)**: `/shared/hackathon/datasets/xray_fl_datasets_preprocessed_128/`
- Preprocessed (224x224): `/shared/hackathon/datasets/xray_fl_datasets_preprocessed_224/`

## 🎯 Expected Results

### Baseline (ResNet18, no pretrained)
- **AUROC**: ~0.70-0.75

### Our Model (EfficientNet B0, pretrained)
- **Expected AUROC**: ~0.75-0.82
- **Best case AUROC**: >0.85 (with tuning)

## 🔧 Hyperparameter Tuning Ideas

Try different configurations:

```bash
# Lower learning rate (recommended for pretrained model)
# Edit pyproject.toml: lr = 0.001

# More local epochs
# Edit pyproject.toml: local-epochs = 2

# Larger images (if time allows)
# Edit pyproject.toml: image-size = 224

# Submit multiple experiments
./submit-job.sh "flwr run . cluster --stream" --gpu --name exp_lr0001_epochs2
./submit-job.sh "flwr run . cluster --stream" --gpu --name exp_lr001_epochs1
./submit-job.sh "flwr run . cluster --stream" --gpu --name exp_lr01_epochs1
```

## 🏥 Hospital Data Distribution

### Hospital A (42,093 train, 5,490 eval)
- Elderly males (60+)
- AP view dominant
- Fluid-related conditions

### Hospital B (21,753 train, 2,860 eval)
- Younger females (20-65)
- PA view dominant
- Nodules, masses, pneumothorax

### Hospital C (20,594 train, 2,730 eval)
- Mixed demographics
- PA view preferred
- Rare conditions

## ⚡ Resource Limits

Per job:
- **1 GPU** (AMD MI300X)
- **32GB RAM**
- **20 minutes** runtime
- **Max 4 concurrent jobs** per team

## 📈 Timeline

| Time | Action |
|------|--------|
| **Now** | Upload template to server |
| **+5 min** | Install dependencies, configure |
| **+10 min** | Submit training job |
| **+30 min** | Training completes (20 min job) |
| **+35 min** | Evaluate best model |
| **+40 min** | Review results, tune if time allows |

**Deadline: Saturday 15:30**

## 🐛 Troubleshooting

### Job Fails Immediately
- Check logs: `tail -f ~/logs/YOUR_JOB_NAME_*.out`
- Verify DATASET_DIR is set correctly
- Make sure dependencies are installed: `pip install -e .`

### Out of Memory
- Reduce image-size from 224 to 128
- Reduce batch size in `task.py`

### Job Timeout at 20 Minutes
- Reduce num-server-rounds (e.g., 100 → 20)
- Reduce local-epochs (e.g., 2 → 1)

### W&B Not Logging
- Set environment variables or hardcode in `server_app.py`
- Training will work without W&B, you just won't see live metrics

## ✅ Success Criteria

- ✅ Training completes without errors
- ✅ Model saved to ~/models/
- ✅ AUROC > 0.70 (baseline)
- 🎯 AUROC > 0.75 (good)
- 🏆 AUROC > 0.80 (excellent)

---

**Good luck! 🚀**
