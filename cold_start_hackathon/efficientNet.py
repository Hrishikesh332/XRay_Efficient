import math
import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from tqdm import tqdm

hospital_datasets = {}  


class Net(nn.Module):


    def __init__(self):
        super(Net, self).__init__()
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Architecture - out_channels=32, kernel_size=3, stride=2, padding=1, bias=False
        original_conv = self.model.features[0][0]
        
        new_conv = nn.Conv2d(
            in_channels=1,  # Grayscale input
            out_channels=original_conv.out_channels,  
            kernel_size=original_conv.kernel_size,  
            stride=original_conv.stride,  
            padding=original_conv.padding,  
            bias=original_conv.bias is not None,  
        )
        

        with torch.no_grad():

            # Shape - (out_channels, 1, kernel_h, kernel_w) from (out_channels, 3, kernel_h, kernel_w)
            new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data.clone()
            else:

                nn.init.zeros_(new_conv.bias) if new_conv.bias is not None else None
        
        self.model.features[0][0] = new_conv
        
        # EfficientNet-B0's classifier structure: [Dropout(p=0.2), Linear(1280, num_classes)]

        in_features = self.model.classifier[1].in_features 
        self.model.classifier[1] = nn.Linear(in_features, 1)
        
        # Use He initialization (Kaiming) which works better with ReLU-based activations
        nn.init.kaiming_uniform_(self.model.classifier[1].weight, a=math.sqrt(5))
        if self.model.classifier[1].bias is not None:

            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.model.classifier[1].weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.01

            nn.init.uniform_(self.model.classifier[1].bias, -bound * 0.5, bound * 0.5)

    def forward(self, x):
        """
        
        Args:
            x: Input tensor of shape (batch_size, 1, height, width)

        """
        output = self.model(x)
        return output.view(-1, 1) if output.dim() == 1 else output


def collate_preprocessed(batch):

    result = {}
    for key in batch[0].keys():
        if key in ["x", "y"]:

            arrays = [np.array(item[key]) for item in batch]
            result[key] = torch.from_numpy(np.stack(arrays))
        else:
            result[key] = [item[key] for item in batch]
    return result


def load_data(
    dataset_name: str,
    split_name: str,
    image_size: int = 128,
    batch_size: int = 32,  # Increased from 16 to 32 for faster training
):
   
    dataset_dir = os.environ["DATASET_DIR"]

    # Use preprocessed dataset based on image_size
    cache_key = f"{dataset_name}_{split_name}_{image_size}"
    dataset_path = f"{dataset_dir}/xray_fl_datasets_preprocessed_{image_size}/{dataset_name}"

    # Load and cache dataset
    global hospital_datasets
    if cache_key not in hospital_datasets:
        full_dataset = load_from_disk(dataset_path)
        hospital_datasets[cache_key] = full_dataset[split_name]
        print(f"Loaded {dataset_path}/{split_name}")

    data = hospital_datasets[cache_key]
    shuffle = (split_name == "train")  # shuffle only for training splits
    

    dataloader = DataLoader(
        data, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=2,  # Reduced from 4 to 2 to prevent shared memory exhaustion
        collate_fn=collate_preprocessed,
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=1,  # Reduced from 2 to 1 to save shared memory
        persistent_workers=False,  # Disabled to reduce shared memory usage
        drop_last=(split_name == "train")  # Drop incomplete batch for training stability
    )
    return dataloader


def train(net, trainloader, epochs, lr, device):

    net.to(device)
    
    # Ensure epochs is integer (fixes float type error)
    epochs = int(epochs) if epochs >= 1.0 else 1
    
    # Use differential learning rates for faster fine-tuning
    # Higher LR for classifier (new layer), lower for pretrained backbone
    backbone_params = []
    classifier_params = []
    first_conv_params = []
    
    for name, param in net.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        elif 'features.0.0' in name:  # First conv layer (adapted for grayscale)
            first_conv_params.append(param)
        else:
            backbone_params.append(param)
    
    # Optimized differential learning rates - balanced for stable convergence
    # Balanced rates to prevent oscillation while maintaining fast learning
    optimizer = torch.optim.AdamW([
        {'params': classifier_params, 'lr': lr * 12.0},  # Balanced at 12x for stable classifier learning
        {'params': first_conv_params, 'lr': lr * 2.5},    # Balanced at 2.5x for stable adaptation
        {'params': backbone_params, 'lr': lr * 0.12}      # Balanced at 0.12x for stable fine-tuning
    ], weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)  
    

    steps_per_epoch = len(trainloader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * 0.15))  # Balanced 15% warmup for steady early learning
    
    # Use cosine annealing with warmup for better convergence
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup - gradual for stability
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay with balanced decay rate
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # Balanced decay - maintain reasonable learning rate
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress * 0.75)))  # Balanced cosine decay
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Better class weight calculation with more aggressive balancing
    # Calculate from multiple batches for better estimate
    try:
        all_pos = 0
        all_neg = 0
        sample_iter = iter(trainloader)
        # Sample from first 5 batches for better estimate
        for _ in range(min(5, len(trainloader))):
            try:
                batch = next(sample_iter)
                y_sample = batch["y"]
                if y_sample.dim() > 1:
                    y_sample = y_sample.squeeze()
                all_pos += (y_sample == 1).sum().item()
                all_neg += (y_sample == 0).sum().item()
            except StopIteration:
                break
        
        if all_pos > 0 and all_neg > 0:
            # Balanced class weighting to handle class imbalance without overcorrection
            pos_weight = torch.tensor([all_neg / all_pos]).to(device)
            # Balanced clamping to prevent extreme weights
            pos_weight = torch.clamp(pos_weight, 0.5, 4.0)  # Balanced range for stable learning
        else:
            pos_weight = torch.tensor([1.0]).to(device)
    except:
        pos_weight = torch.tensor([1.0]).to(device)
    

    class FocalLoss(nn.Module):
        def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma  # Moderate gamma for balanced learning
            self.pos_weight = pos_weight
            
        def forward(self, inputs, targets):
            # BCE with logits
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                inputs, targets, pos_weight=self.pos_weight, reduction='none'
            )
            # Get probabilities (stable calculation)
            pt = torch.sigmoid(inputs) * targets + (1 - torch.sigmoid(inputs)) * (1 - targets)
            # Focal term - moderate gamma focuses on hard examples
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
            return focal_loss.mean()
    
    # Use focal loss for better convergence, fallback to BCE if needed
    try:
        criterion = FocalLoss(alpha=1.0, gamma=2.0, pos_weight=pos_weight).to(device)
    except:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    
    # CRITICAL: Ensure model is in float32 to avoid dtype mismatches with mixed precision
    net = net.float()
    net.train()
    
    # Mixed precision training for 2x speedup on GPU
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = device.type == 'cuda' and scaler is not None
    
    running_loss = 0.0
    total_batches = 0
    
    # Check if wandb is initialized
    use_wandb = wandb.run is not None
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        

        progress_bar = trainloader if epochs == 1 else tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", mininterval=1.0)
        
        for batch in progress_bar:

            x = batch["x"].to(device, non_blocking=True).float()
            y = batch["y"].to(device, non_blocking=True)
            
            # Ensure y has correct shape for BCEWithLogitsLoss
            if y.dim() == 1:
                y = y.unsqueeze(-1).float()
            elif y.dim() == 2 and y.size(1) == 1:
                y = y.float()
            else:
                y = y.float()
            
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = net(x)
                    # Ensure outputs and y have compatible shapes
                    if outputs.dim() == 2 and outputs.size(1) == 1:
                        outputs = outputs.squeeze(-1) if y.dim() == 1 else outputs
                    elif outputs.dim() == 1 and y.dim() == 2:
                        outputs = outputs.unsqueeze(-1)
                    loss = criterion(outputs, y)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                current_step = total_batches
                if current_step < warmup_steps:
                    max_norm = 3.0  # Higher clipping during warmup for faster learning
                else:
                    max_norm = 1.5  # Lower clipping after warmup for stability
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=max_norm)
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
            else:
                outputs = net(x)
                # Ensure outputs and y have compatible shapes
                if outputs.dim() == 2 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(-1) if y.dim() == 1 else outputs
                elif outputs.dim() == 1 and y.dim() == 2:
                    outputs = outputs.unsqueeze(-1)
                loss = criterion(outputs, y)
                loss.backward()
                # Adaptive gradient clipping
                current_step = total_batches
                if current_step < warmup_steps:
                    max_norm = 3.0
                else:
                    max_norm = 1.5
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=max_norm)
                optimizer.step()
                # CRITICAL: Update scheduler AFTER optimizer.step() (PyTorch requirement)
                scheduler.step()
            
            epoch_loss += loss.item()
            total_batches += 1
            batch_count += 1
            
            # Reduced wandb logging frequency to minimize overhead
            if use_wandb and batch_count % 200 == 0:  # Log every 200 batches instead of 50
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/learning_rate": current_lr
                })
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        running_loss += epoch_loss
        
        # Log epoch loss to wandb if initialized
        if use_wandb:
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                "train/epoch_loss": avg_epoch_loss,
                "train/epoch": epoch + 1,
                "train/learning_rate": current_lr,
                "train/classifier_lr": optimizer.param_groups[0]['lr'],
                "train/backbone_lr": optimizer.param_groups[2]['lr']
            })
    
    avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss


def test(net, testloader, device):

    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    net.eval()
    total_loss = 0.0

    all_probs = []
    all_predictions = []
    all_labels = []
    

    use_wandb = wandb.run is not None

    use_amp = device.type == 'cuda'
    
    with torch.no_grad():
        for batch in testloader:

            x = batch["x"].to(device, non_blocking=True).float()
            y = batch["y"].to(device, non_blocking=True)
            
            # Ensure y has correct shape
            if y.dim() == 1:
                y = y.unsqueeze(-1).float()
            else:
                y = y.float()

            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = net(x)
            else:
                outputs = net(x)
            
            # Ensure outputs and y have compatible shapes
            if outputs.dim() == 2 and outputs.size(1) == 1:
                outputs_for_loss = outputs.squeeze(-1) if y.dim() == 1 else outputs
            elif outputs.dim() == 1 and y.dim() == 2:
                outputs_for_loss = outputs.unsqueeze(-1)
            else:
                outputs_for_loss = outputs
            
            loss = criterion(outputs_for_loss, y)
            total_loss += loss.item()

            outputs_flat = outputs.squeeze(-1) if outputs.dim() > 1 and outputs.size(-1) == 1 else outputs
            outputs_flat = outputs_flat.float()  # Convert from float16 (Half) to float32
            probs = torch.sigmoid(outputs_flat)

            threshold = 0.5
            predictions = (probs > threshold).float()

            # Optimized storage - convert to float32 before numpy to avoid dtype issues
            all_probs.extend(probs.cpu().float().numpy().flatten())
            all_predictions.extend(predictions.cpu().float().numpy().flatten())
            all_labels.extend(y.squeeze(-1).cpu().float().numpy() if y.dim() > 1 else y.cpu().float().numpy())

    avg_loss = total_loss / len(testloader) if len(testloader) > 0 else 0.0

    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate confusion matrix components
    tp = int(np.sum((all_predictions == 1) & (all_labels == 1)))
    tn = int(np.sum((all_predictions == 0) & (all_labels == 0)))
    fp = int(np.sum((all_predictions == 1) & (all_labels == 0)))
    fn = int(np.sum((all_predictions == 0) & (all_labels == 1)))
    
    # Calculate additional metrics for wandb logging
    if use_wandb:
        from sklearn.metrics import roc_auc_score, accuracy_score
        accuracy = accuracy_score(all_labels, all_predictions)
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auroc = 0.0
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        wandb.log({
            "eval/loss": avg_loss,
            "eval/accuracy": accuracy,
            "eval/auroc": auroc,
            "eval/sensitivity": sensitivity,
            "eval/specificity": specificity,
            "eval/precision": precision,
            "eval/f1": f1,
        })

    return avg_loss, tp, tn, fp, fn, all_probs, all_labels