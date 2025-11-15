# Federated Chest X-Ray Classification

This project implements federated learning for binary chest x-ray classification across 3 hospital silos using Flower framework and EfficientNetB0.

## Challenge Details

- **Task** - Binary classification (No Finding vs Any Finding)
- **Metric** - AUROC
- **Dataset** - NIH Chest X-Ray (non-IID split across 3 hospitals)
- **Model** - EfficientNetB0 with custom Layer

## Project Structure

```
coldstart/
├── pyproject.toml          # Project configuration and Flower federation settings
├── README.md               # This file
├── coldstart/
│   ├── __init__.py
│   ├── models.py           # EfficientNetB0 model architecture
│   ├── dataset.py          # Dataset loading and preprocessing
│   ├── client_app.py       # Federated client logic
│   ├── server_app.py       # Federated server and strategy
│   └── utils.py            # Utility functions (metrics, etc.)
└── requirements.txt        # Python dependencies
```

## Major Improvement over the EfficientNet B0 model

Applied Differential learning rates - classifier (12x), first conv (2.5x), backbone (0.12x)

Learning rate scheduler - 15% linear warmup + cosine decay (min_lr=0.05)

Optimizer - AdamW with weight_decay=1e-4, amsgrad=True

Focal Loss (gamma=2.0, alpha=1.0) with class weighting

Adaptive gradient clipping - 3.0 (warmup) → 1.5 (after)

Mixed precision training (FP16) with GradScaler

Batch size increased from 16 to 32

## Metrics

### AUROC

![My Image](https://drive.google.com/uc?export=view&id=1XgMgaekVTz5hwmmzDErkm7YLhDVBe8tL)


### Training and Loss

![My Image](https://drive.google.com/uc?export=view&id=182TDslYJOcmSjPWH40nMYYKvFFVYgboD)

## Local Development

1. Install dependencies:
```bash
pip install -e .
```

2. Run simulation locally:
```bash
flwr run . local-simulation
```

## Cluster Deployment

1. Sync to cluster:
```bash
scp -P 32605 -r coldstart/ team04@129.212.178.168:~/
```

2. SSH into cluster:
```bash
ssh team04@129.212.178.168 -p 32605
```

3. Submit GPU job:
```bash
cd ~/coldstart
./submit-job.sh "flwr run . cluster-gpu" --gpu --name xray-fed
```

4. Monitor logs:
```bash
tail -f ~/logs/job*_xray-fed.out
```

## Model Architecture

- **Backbone**: EfficientNetB0 (pretrained on ImageNet)
- **Input**: 128x128 grayscale images
- **Output**: Binary classification (sigmoid activation)
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam with learning rate scheduling

## Federated Strategy

- **Algorithm** - FedAvg with weighted averaging
- **Aggregation** - Based on number of samples per client
- **Client sampling** - All 3 hospitals per round

## Data Augmentation

- Random horizontal flip
- Random rotation (±10 degrees)
- Random brightness/contrast adjustment
- Normalization (ImageNet stats)
