"""Evaluation script.

If you can run this, so can we during final evaluation!

Usage:
  1. Update MODEL_PATH below with your best model filename
  2. Run in SLURM: ./submit-job.sh "python evaluate.py" --gpu
"""

import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from cold_start_hackathon.task import Net, load_data, test

# Model path: update with your best model filename from ~/models/
# Example: MODEL_PATH = f"/home/team00/models/job_123456_round5_auroc8234.pt"
MODEL_PATH = f"/home/YOUR_TEAM_NAME/models/YOUR_MODEL_FILE.pt"
DATASET_DIR = os.environ["DATASET_DIR"]


def evaluate_split(model, dataset_name, split_name, device):
    """Evaluate on any dataset split and return predictions."""
    loader = load_data(dataset_name, split_name, batch_size=32)
    _, _, _, _, _, probs, labels = test(model, loader, device)
    return probs, labels


def main():
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    print(f"\nLoading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please update MODEL_PATH in evaluate.py with your model filename.")
        return 1

    # Load model
    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}.")

    # Evaluate
    print("\nEvaluating...")
    datasets_to_test = [
        ("Hospital A", "HospitalA", "eval"),
        ("Hospital B", "HospitalB", "eval"),
        ("Hospital C", "HospitalC", "eval"),
        ("Test A", "Test", "test_A"),
        ("Test B", "Test", "test_B"),
        ("Test C", "Test", "test_C"),
        ("Test D (OOD)", "Test", "test_D"),
    ]

    # Collect all predictions
    hospital_predictions = {}
    test_predictions = {}
    for display_name, dataset_name, split_name in datasets_to_test:
        try:
            probs, labels = evaluate_split(model, dataset_name, split_name, device)
            n = len(labels)

            # Compute per-dataset AUROC for display
            auroc = roc_auc_score(labels, probs)
            print(f"  {display_name:<15} AUROC: {auroc:.4f} (n={n})")

            # Store predictions for aggregated AUROC calculation
            if display_name.startswith("Hospital"):
                hospital_predictions[display_name] = (probs, labels)
            elif display_name.startswith("Test"):
                test_predictions[display_name] = (probs, labels)
        except FileNotFoundError:
            # Test dataset doesn't exist for participants - skip silently
            pass

    # Eval Average: pool all hospital eval predictions, then compute AUROC
    if hospital_predictions:
        all_probs = np.concatenate([probs for probs, _ in hospital_predictions.values()])
        all_labels = np.concatenate([labels for _, labels in hospital_predictions.values()])
        eval_auroc = roc_auc_score(all_labels, all_probs)
        print(f"  {'Eval Avg':<15} AUROC: {eval_auroc:.4f}")

    # Test Average: pool all test predictions, then compute AUROC
    if test_predictions:
        all_probs = np.concatenate([probs for probs, _ in test_predictions.values()])
        all_labels = np.concatenate([labels for _, labels in test_predictions.values()])
        test_auroc = roc_auc_score(all_labels, all_probs)
        print(f"  {'Test Avg':<15} AUROC: {test_auroc:.4f}")

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    exit(main())
