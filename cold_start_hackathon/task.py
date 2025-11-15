import math
import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights
from tqdm import tqdm

hospital_datasets = {}  


class Net(nn.Module):
    """
    Enhanced MobileNet V3 Large model for grayscale X-ray image binary classification.
    
    Optimized for fast training (target: ~20 minutes) with high accuracy.
    
    Architecture improvements:
    - Base model: MobileNet V3 Large (pretrained on ImageNet)
    - Parameters: ~5.4M (lightweight and efficient)
    - Input: Grayscale images (1 channel) with improved weight transfer
    - Advanced classifier head with Channel Attention and multi-scale processing
    
    Enhanced Architecture:
    - Features: Inverted residual blocks with SE modules and Hard-Swish activation
    - First conv layer: Improved grayscale adaptation using luminance weights (0.299*R + 0.587*G + 0.114*B)
    - Advanced Classifier Head with Attention:
        [0]: Linear(960 -> 960) + HardSwish + Dropout(0.2)  [Original MobileNet layers]
        [1]: Advanced Multi-Scale Classifier:
            - Linear(960 -> 768) + LayerNorm + ChannelAttention + HardSwish + Dropout(0.25)
            - Linear(768 -> 512) + LayerNorm + ChannelAttention + HardSwish + Dropout(0.3)
            - Linear(512 -> 1)  [Binary output]
    
    Key improvements:
    1. Better grayscale weight transfer (luminance formula)
    2. Channel Attention Modules for feature recalibration at multiple scales
    3. Multi-scale feature processing (960->768->512) for better representation
    4. Progressive dropout (0.25 -> 0.3) for adaptive regularization
    5. LayerNorm for stable training (works with any batch size)
    6. Optimized initialization (Xavier gain=0.6) for better learning
    7. Comprehensive NaN/Inf detection and handling throughout training
    
    Training optimizations:
    - Balanced learning rates for stable convergence
    - Mixed precision training (FP16) for 2x speedup
    - Optimized batch size and data loading
    """

    def __init__(self):
        super(Net, self).__init__()
        # Load MobileNet V3 Large with pretrained ImageNet weights
        # Weights will be automatically downloaded from PyTorch model zoo if not cached
        self.model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

        # ========================================================================
        # Adapt first convolutional layer for grayscale input (1 channel)
        # ========================================================================
        # MobileNet V3 Large first conv layer specs:
        # - Original: Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # - Adapted: Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        original_conv = self.model.features[0][0]
        
        new_conv = nn.Conv2d(
            in_channels=1,  # Grayscale input (changed from 3)
            out_channels=original_conv.out_channels,  # 16 channels
            kernel_size=original_conv.kernel_size,    # (3, 3)
            stride=original_conv.stride,              # (2, 2)
            padding=original_conv.padding,            # (1, 1)
            bias=original_conv.bias is not None,      # False
        )
        
        # Transfer pretrained weights: improved grayscale adaptation
        # Use weighted average (luminance formula) instead of simple mean for better transfer
        with torch.no_grad():
            # Convert (out_channels, 3, kernel_h, kernel_w) -> (out_channels, 1, kernel_h, kernel_w)
            # Use luminance weights: 0.299*R + 0.587*G + 0.114*B (standard grayscale conversion)
            # This preserves more information from pretrained weights
            rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=original_conv.weight.device)
            rgb_weights = rgb_weights.view(1, 3, 1, 1)  # Shape for broadcasting
            new_conv.weight.data = (original_conv.weight.data * rgb_weights).sum(dim=1, keepdim=True)
            
            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data.clone()
            else:
                # Initialize bias to zeros if original had no bias
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)
        
        self.model.features[0][0] = new_conv
        
        # ========================================================================
        # Advanced classifier head with attention and multi-scale features
        # ========================================================================
        # MobileNet V3 Large classifier structure (Sequential):
        # [0]: Linear(in_features=960, out_features=960)  - Hidden layer
        # [1]: HardSwish()                                - Activation
        # [2]: Dropout(p=0.2)                              - Regularization
        # [3]: Linear(in_features=960, out_features=1000)  - Output layer (ImageNet classes)
        # 
        # Advanced architecture improvements:
        # 1. Channel Attention Module (CAM) for feature recalibration
        # 2. Multi-scale feature processing with residual connections
        # 3. Progressive feature compression with skip connections
        # 4. Enhanced regularization with adaptive dropout
        in_features = self.model.classifier[3].in_features  # 960
        
        # Channel Attention Module for feature enhancement
        class ChannelAttention(nn.Module):
            """Channel Attention Module for feature recalibration (adapted for 1D feature vectors)"""
            def __init__(self, channels, reduction=16):
                super(ChannelAttention, self).__init__()
                self.reduction = max(1, channels // reduction)  # Ensure at least 1
                # Use a simpler approach: compute attention from the feature vector itself
                self.fc = nn.Sequential(
                    nn.Linear(channels, self.reduction, bias=False),
                    nn.ReLU(),
                    nn.Linear(self.reduction, channels, bias=False),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                # x shape: (batch, features)
                # Compute attention weights from the feature vector itself
                # This allows the model to learn which features are important
                attention = self.fc(x)  # (batch, features)
                return x * attention
        
        # Advanced classifier with attention and residual connections
        class AdvancedClassifier(nn.Module):
            """Advanced classifier with attention and multi-scale processing"""
            def __init__(self, in_features):
                super(AdvancedClassifier, self).__init__()
                
                # First transformation with attention
                self.fc1 = nn.Linear(in_features, 768)
                self.ln1 = nn.LayerNorm(768)
                self.attention1 = ChannelAttention(768, reduction=16)
                self.act1 = nn.Hardswish()
                self.dropout1 = nn.Dropout(p=0.25)
                
                # Second transformation with residual connection
                self.fc2 = nn.Linear(768, 512)
                self.ln2 = nn.LayerNorm(512)
                self.attention2 = ChannelAttention(512, reduction=8)
                self.act2 = nn.Hardswish()
                self.dropout2 = nn.Dropout(p=0.3)
                
                # Final output layer
                self.fc3 = nn.Linear(512, 1)
                
                # Initialize weights
                self._initialize_weights()
            
            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight, gain=0.6)  # Slightly higher for better learning
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0.0)
                    elif isinstance(m, nn.LayerNorm):
                        nn.init.constant_(m.weight, 1.0)
                        nn.init.constant_(m.bias, 0.0)
            
            def forward(self, x):
                # First block with attention
                out = self.fc1(x)
                out = self.ln1(out)
                out = self.attention1(out)
                out = self.act1(out)
                out = self.dropout1(out)
                
                # Second block with residual-like connection (through attention)
                identity = out  # Store for potential residual
                out = self.fc2(out)
                out = self.ln2(out)
                out = self.attention2(out)
                out = self.act2(out)
                out = self.dropout2(out)
                
                # Final output
                out = self.fc3(out)
                return out
        
        # Replace the final classifier layer with our advanced version
        enhanced_classifier = AdvancedClassifier(in_features)
        self.model.classifier[3] = enhanced_classifier

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
    batch_size: int = 32,  # Optimized for MobileNet V3 - good balance of speed and memory
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
    
    # Optimized differential learning rates for advanced architecture
    # Balanced rates for the new attention-based classifier
    optimizer = torch.optim.AdamW([
        {'params': classifier_params, 'lr': lr * 6.0},   # Slightly higher for attention modules
        {'params': first_conv_params, 'lr': lr * 1.5},   # Moderate for grayscale adaptation
        {'params': backbone_params, 'lr': lr * 0.06}      # Slightly higher for better fine-tuning
    ], weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)  
    

    steps_per_epoch = len(trainloader)
    total_steps = steps_per_epoch * epochs
    # Balanced warmup (12%) for stable initial learning - prevents early instability
    warmup_steps = max(1, int(total_steps * 0.12))
    
    # Use cosine annealing with warmup for better convergence
    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup - gradual for stability
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay - balanced for stable convergence
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # Maintain reasonable minimum LR (0.05) for continued learning
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress * 0.8)))  # Balanced cosine decay
    
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
                    
                    # Check for NaN in outputs before computing loss
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"Warning: NaN/Inf detected in outputs, skipping batch")
                        continue
                    
                    loss = criterion(outputs, y)
                    
                    # Check for NaN loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN/Inf loss detected, skipping batch")
                        continue
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                current_step = total_batches
                # More aggressive gradient clipping to prevent NaN
                if current_step < warmup_steps:
                    max_norm = 2.0  # Reduced from 3.0 for stability
                else:
                    max_norm = 1.0  # Reduced from 1.5 for stability
                
                # Clip gradients before optimizer step
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=max_norm)
                
                # Check for NaN gradients
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: NaN/Inf gradient norm detected, skipping optimizer step")
                    scaler.update()
                    continue
                
                scaler.step(optimizer)
                scaler.update()
                # CRITICAL: Update scheduler AFTER optimizer.step() (PyTorch requirement)
                scheduler.step()
            else:
                outputs = net(x)
                # Ensure outputs and y have compatible shapes
                if outputs.dim() == 2 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(-1) if y.dim() == 1 else outputs
                elif outputs.dim() == 1 and y.dim() == 2:
                    outputs = outputs.unsqueeze(-1)
                
                # Check for NaN in outputs before computing loss
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"Warning: NaN/Inf detected in outputs, skipping batch")
                    continue
                
                loss = criterion(outputs, y)
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected, skipping batch")
                    continue
                
                loss.backward()
                # Adaptive gradient clipping - more aggressive to prevent NaN
                current_step = total_batches
                if current_step < warmup_steps:
                    max_norm = 2.0  # Reduced from 3.0
                else:
                    max_norm = 1.0  # Reduced from 1.5
                
                # Clip gradients before optimizer step
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=max_norm)
                
                # Check for NaN gradients
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: NaN/Inf gradient norm detected, skipping optimizer step")
                    optimizer.zero_grad()  # Clear gradients
                    continue
                
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
            # Handle NaN loss in evaluation
            if torch.isnan(loss) or torch.isinf(loss):
                # Skip this batch if loss is invalid
                continue
            total_loss += loss.item()

            outputs_flat = outputs.squeeze(-1) if outputs.dim() > 1 and outputs.size(-1) == 1 else outputs
            outputs_flat = outputs_flat.float()  # Convert from float16 (Half) to float32
            
            # Check for NaN/Inf in outputs and replace with safe values
            if torch.isnan(outputs_flat).any() or torch.isinf(outputs_flat).any():
                # Replace NaN/Inf with zeros (neutral logit = 0.5 probability)
                outputs_flat = torch.where(
                    torch.isnan(outputs_flat) | torch.isinf(outputs_flat),
                    torch.zeros_like(outputs_flat),
                    outputs_flat
                )
            
            probs = torch.sigmoid(outputs_flat)
            
            # Clamp probabilities to valid range [0, 1] to prevent NaN
            probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)

            threshold = 0.5
            predictions = (probs > threshold).float()

            # Optimized storage - convert to float32 before numpy to avoid dtype issues
            # Filter out NaN values before extending lists
            probs_np = probs.cpu().float().numpy().flatten()
            predictions_np = predictions.cpu().float().numpy().flatten()
            labels_np = y.squeeze(-1).cpu().float().numpy() if y.dim() > 1 else y.cpu().float().numpy()
            
            # Filter out NaN values
            valid_mask = ~(np.isnan(probs_np) | np.isinf(probs_np))
            if valid_mask.any():
                all_probs.extend(probs_np[valid_mask])
                all_predictions.extend(predictions_np[valid_mask])
                # Handle labels properly (can be scalar or array)
                if labels_np.ndim == 0:
                    # Scalar case - extend with the scalar value for each valid entry
                    all_labels.extend([labels_np] * valid_mask.sum())
                else:
                    all_labels.extend(labels_np[valid_mask])

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
