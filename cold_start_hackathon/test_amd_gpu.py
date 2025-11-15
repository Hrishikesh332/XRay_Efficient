"""
Quick test script to verify AMD GPU and mixed precision support.

Run this to check if your AMD GPU is properly configured:
    python test_amd_gpu.py
"""

import torch
from torch.cuda.amp import autocast, GradScaler

def test_gpu_setup():
    """Test GPU availability and mixed precision support."""
    print("=" * 60)
    print("GPU Setup Test")
    print("=" * 60)
    
    # Check PyTorch version
    print(f"\n📦 PyTorch version: {torch.__version__}")
    
    # Check CUDA/ROCm availability
    cuda_available = torch.cuda.is_available()
    print(f"🎮 CUDA/ROCm available: {cuda_available}")
    
    if not cuda_available:
        print("\n❌ GPU not available!")
        print("   Possible issues:")
        print("   1. PyTorch not built with ROCm support (for AMD)")
        print("   2. GPU drivers not installed")
        print("   3. ROCm not properly configured")
        print("\n   Solution: Install PyTorch with ROCm support")
        print("   See: https://pytorch.org/get-started/locally/")
        return False
    
    # Get device info
    device_count = torch.cuda.device_count()
    print(f"🎯 GPU count: {device_count}")
    
    for i in range(device_count):
        device = torch.device(f"cuda:{i}")
        print(f"\n📱 GPU {i}:")
        print(f"   Name: {torch.cuda.get_device_name(i)}")
        
        try:
            props = torch.cuda.get_device_properties(i)
            print(f"   Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
        except:
            print("   (Some properties not available)")
    
    # Test mixed precision
    print("\n" + "=" * 60)
    print("Testing Mixed Precision")
    print("=" * 60)
    
    try:
        device = torch.device("cuda:0")
        
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 10)
        ).to(device)
        
        # Create dummy input
        x = torch.randn(4, 3, 224, 224).to(device)
        target = torch.randint(0, 10, (4,)).to(device)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = GradScaler()
        
        # Test mixed precision forward/backward
        optimizer.zero_grad()
        with autocast():
            output = model(x)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("\n✅ Mixed precision test PASSED!")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Mixed precision test FAILED!")
        print(f"   Error: {str(e)}")
        print("\n   This might indicate:")
        print("   1. ROCm version too old (need 5.0+)")
        print("   2. GPU doesn't support FP16 operations")
        print("   3. PyTorch ROCm build issue")
        return False

def test_model_loading():
    """Test if we can load the model architecture."""
    print("\n" + "=" * 60)
    print("Testing Model Architecture")
    print("=" * 60)
    
    try:
        from torchvision import models
        from torchvision.models import MobileNet_V3_Small_Weights
        
        # Test MobileNetV3 loading
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        print("✅ MobileNetV3-Small loaded successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model = model.to(device)
            x = torch.randn(2, 3, 224, 224).to(device)
            
            with torch.no_grad():
                output = model(x)
            print(f"   Forward pass successful: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test FAILED!")
        print(f"   Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n🚀 Starting GPU compatibility tests...\n")
    
    gpu_ok = test_gpu_setup()
    model_ok = test_model_loading()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if gpu_ok and model_ok:
        print("\n✅ All tests passed! Your setup is ready for training.")
        print("\nYou can now use:")
        print("  - Mixed precision training (1.5-2x speedup)")
        print("  - MobileNetV3 architecture (2-3x speedup)")
        print("  - All optimizations from the guide")
    elif gpu_ok:
        print("\n⚠️  GPU works, but model loading failed.")
        print("   Check torchvision installation.")
    else:
        print("\n❌ GPU setup needs attention.")
        print("   See AMD_GPU_GUIDE.md for troubleshooting.")
    
    print()

