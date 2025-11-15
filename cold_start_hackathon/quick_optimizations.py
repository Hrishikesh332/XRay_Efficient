"""
Quick optimization snippets to add to your existing task.py

These are minimal changes that can significantly speed up training.
Copy and paste these into your existing code.
"""

# ============================================================================
# OPTIMIZATION 1: Mixed Precision Training (1.5-2x speedup)
# ============================================================================
# Add this import at the top:
from torch.cuda.amp import autocast, GradScaler

# Modify the train() function:
def train_optimized_mixed_precision(net, trainloader, epochs, lr, device):
    """Train with mixed precision - 1.5-2x faster on GPU."""
    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    
    # Enable mixed precision if GPU available
    # Works on both NVIDIA (CUDA) and AMD (ROCm) GPUs
    # Note: For AMD GPUs, ensure PyTorch is built with ROCm support
    use_amp = device.type == 'cuda'  # 'cuda' works for both NVIDIA and AMD GPUs with ROCm
    scaler = GradScaler() if use_amp else None
    
    net.train()
    running_loss = 0.0
    total_batches = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            
            if y.dim() == 1:
                y = y.unsqueeze(-1).float()
            else:
                y = y.float()
            
            optimizer.zero_grad()
            
            if use_amp:
                # Mixed precision forward pass
                with autocast():
                    outputs = net(x)
                    if outputs.dim() == 2 and outputs.size(1) == 1:
                        outputs = outputs.squeeze(-1) if y.dim() == 1 else outputs
                    elif outputs.dim() == 1 and y.dim() == 2:
                        outputs = outputs.unsqueeze(-1)
                    loss = criterion(outputs, y)
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision (CPU fallback)
                outputs = net(x)
                if outputs.dim() == 2 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(-1) if y.dim() == 1 else outputs
                elif outputs.dim() == 1 and y.dim() == 2:
                    outputs = outputs.unsqueeze(-1)
                loss = criterion(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            total_batches += 1
        
        running_loss += epoch_loss
    
    avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss


# ============================================================================
# OPTIMIZATION 2: Learning Rate Scheduling (faster convergence)
# ============================================================================
# Add this to train() function after creating optimizer:
def train_with_scheduler(net, trainloader, epochs, lr, device):
    """Train with learning rate scheduling for faster convergence."""
    # ... existing setup code ...
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs * len(trainloader)  # Total number of steps
    )
    
    # In training loop, after optimizer.step():
    optimizer.step()
    scheduler.step()  # Update learning rate
    # ... rest of code ...


# ============================================================================
# OPTIMIZATION 3: Larger Batch Sizes (if using mixed precision)
# ============================================================================
# In load_data() function, you can increase batch_size:
# - With FP16: batch_size=32 or 64 (instead of 16)
# - This allows more parallel processing
# - Update: batch_size: int = 32,  # Increased from 16


# ============================================================================
# OPTIMIZATION 4: Progressive Resizing Strategy
# ============================================================================
# Train on smaller images first, then fine-tune on larger:
# 
# Round 1-5:  image_size=128, epochs=3
# Round 6-10: image_size=224, epochs=2
#
# This reduces total compute while improving final accuracy


# ============================================================================
# OPTIMIZATION 5: Selective Layer Freezing (faster initial training)
# ============================================================================
def freeze_backbone_layers(net, freeze_ratio=0.7):
    """Freeze early layers to speed up initial training.
    
    Args:
        net: The neural network model
        freeze_ratio: Fraction of layers to freeze (0.0-1.0)
    """
    total_layers = len(list(net.model.features))
    freeze_until = int(total_layers * freeze_ratio)
    
    # Freeze early feature layers
    for i, layer in enumerate(net.model.features):
        if i < freeze_until:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True
    
    # Always train classifier
    for param in net.model.classifier.parameters():
        param.requires_grad = True
    
    return net


# ============================================================================
# COMBINED: All Optimizations Together
# ============================================================================
def train_fully_optimized(net, trainloader, epochs, lr, device, freeze_early=True):
    """Fully optimized training with all speed improvements."""
    net.to(device)
    
    # Optionally freeze early layers for faster initial training
    if freeze_early:
        net = freeze_backbone_layers(net, freeze_ratio=0.6)
    
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(trainloader)
    )
    
    # Mixed precision
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    net.train()
    running_loss = 0.0
    total_batches = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            
            if y.dim() == 1:
                y = y.unsqueeze(-1).float()
            else:
                y = y.float()
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    outputs = net(x)
                    if outputs.dim() == 2 and outputs.size(1) == 1:
                        outputs = outputs.squeeze(-1) if y.dim() == 1 else outputs
                    elif outputs.dim() == 1 and y.dim() == 2:
                        outputs = outputs.unsqueeze(-1)
                    loss = criterion(outputs, y)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = net(x)
                if outputs.dim() == 2 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(-1) if y.dim() == 1 else outputs
                elif outputs.dim() == 1 and y.dim() == 2:
                    outputs = outputs.unsqueeze(-1)
                loss = criterion(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            epoch_loss += loss.item()
            total_batches += 1
        
        running_loss += epoch_loss
    
    avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss

