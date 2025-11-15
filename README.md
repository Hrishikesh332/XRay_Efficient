# Federated Chest X-Ray Classification

This project implements federated learning for binary chest x-ray classification across 3 hospital silos using Flower framework and EfficientNetB0.

## Challenge Details

- **Task**: Binary classification (No Finding vs Any Finding)
- **Metric**: AUROC
- **Dataset**: NIH Chest X-Ray (non-IID split across 3 hospitals)
- **Model**: EfficientNetB0 with custom head

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
- **Input**: 224x224 grayscale images
- **Output**: Binary classification (sigmoid activation)
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam with learning rate scheduling

## Federated Strategy

- **Algorithm**: FedAvg with weighted averaging
- **Aggregation**: Based on number of samples per client
- **Rounds**: 50 (configurable in pyproject.toml)
- **Client sampling**: All 3 hospitals per round

## Data Augmentation

- Random horizontal flip
- Random rotation (±10 degrees)
- Random brightness/contrast adjustment
- Normalization (ImageNet stats)
