#!/usr/bin/env python3
"""Train a model on a single hospital's data."""

import argparse
import os
import sys
from pathlib import Path
import json

import torch
import torch.nn as nn
from tqdm import tqdm

# Add coldstart to path
sys.path.insert(0, str(Path(__file__).parent))

from coldstart.models import get_model, get_parameters
from coldstart.dataset import load_data
from coldstart.utils import train_epoch, evaluate, get_optimizer, get_scheduler


def train_hospital_model(
    hospital_id: str,
    data_dir: str,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    image_size: int = 224,
    output_dir: str = "models",
    device: str = None
):
    """Train a model on a single hospital's data.

    Args:
        hospital_id: Hospital identifier ('A', 'B', or 'C')
        data_dir: Root directory containing hospital data
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        image_size: Image size (128 or 224)
        output_dir: Directory to save model checkpoints
        device: Device to train on (auto-detect if None)
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    print(f"\n{'='*60}")
    print(f"Training Model for Hospital {hospital_id}")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Image size: {image_size}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    train_loader, eval_loader = load_data(
        data_dir=data_dir,
        hospital_id=hospital_id,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=4
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Eval samples: {len(eval_loader.dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Eval batches: {len(eval_loader)}\n")

    # Create model
    print("Creating model...")
    model = get_model(device=device)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer and scheduler
    optimizer = get_optimizer(model, lr=learning_rate)
    scheduler = get_scheduler(optimizer, num_epochs=num_epochs)

    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'eval_loss': [],
        'eval_accuracy': [],
        'eval_auroc': [],
        'eval_precision': [],
        'eval_recall': [],
        'eval_f1': []
    }

    best_auroc = 0.0
    best_model_path = output_path / f"hospital_{hospital_id}_best.pth"

    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...\n")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # Evaluate
        eval_metrics = evaluate(
            model,
            eval_loader,
            criterion,
            device
        )

        # Update scheduler
        scheduler.step()

        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['eval_loss'].append(eval_metrics['loss'])
        history['eval_accuracy'].append(eval_metrics['accuracy'])
        history['eval_auroc'].append(eval_metrics['auroc'])
        history['eval_precision'].append(eval_metrics['precision'])
        history['eval_recall'].append(eval_metrics['recall'])
        history['eval_f1'].append(eval_metrics['f1'])

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Eval Loss: {eval_metrics['loss']:.4f} | Eval Acc: {eval_metrics['accuracy']:.4f}")
        print(f"AUROC: {eval_metrics['auroc']:.4f} | F1: {eval_metrics['f1']:.4f}")
        print(f"Precision: {eval_metrics['precision']:.4f} | Recall: {eval_metrics['recall']:.4f}")

        # Save best model
        if eval_metrics['auroc'] > best_auroc:
            best_auroc = eval_metrics['auroc']
            print(f"✓ New best AUROC! Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auroc': best_auroc,
                'metrics': eval_metrics,
                'hospital_id': hospital_id
            }, best_model_path)

        print()

    # Save final model
    final_model_path = output_path / f"hospital_{hospital_id}_final.pth"
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'hospital_id': hospital_id
    }, final_model_path)

    # Save training history
    history_path = output_path / f"hospital_{hospital_id}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training Complete for Hospital {hospital_id}")
    print(f"{'='*60}")
    print(f"Best AUROC: {best_auroc:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"History saved to: {history_path}")
    print(f"{'='*60}\n")

    return history, best_auroc


def main():
    parser = argparse.ArgumentParser(description='Train model on single hospital data')
    parser.add_argument('--hospital', type=str, required=True, choices=['A', 'B', 'C'],
                        help='Hospital ID (A, B, or C)')
    parser.add_argument('--data-dir', type=str,
                        default=os.getenv('DATA_DIR', '~/xray-data/xray_fl_datasets_preprocessed_224'),
                        help='Root directory containing hospital data')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--image-size', type=int, default=224, choices=[128, 224],
                        help='Image size')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for model checkpoints')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='Device to train on (auto-detect if not specified)')

    args = parser.parse_args()

    # Expand user path
    data_dir = os.path.expanduser(args.data_dir)

    # Train
    train_hospital_model(
        hospital_id=args.hospital,
        data_dir=data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
