#!/usr/bin/env python3
"""
Explainability Script for MobileNet V3 X-ray Classification Model

This script applies Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize
which regions of an X-ray image the model focuses on when making predictions.

Usage:
    # First, activate the virtual environment:
    source ~/hackathon-venv/bin/activate
    
    # Then run the script:
    python explainability.py --model_path <path_to_model.pt> --image_path <path_to_image> [--output_path <output_path>]
    
    # Or use default model (best AUROC from models/):
    python explainability.py --image_path <path_to_image>

Requirements:
    - torch
    - torchvision
    - numpy
    - matplotlib
    - PIL
    - argparse
    - opencv-python (optional, for better image processing)
"""

import argparse
import os
import sys

# Check if running in virtual environment and provide helpful error message
try:
    import torch
except ImportError:
    print("=" * 70)
    print("ERROR: PyTorch not found!")
    print("=" * 70)
    print("\nPlease activate the virtual environment first:")
    print("  source ~/hackathon-venv/bin/activate")
    print("\nOr if using a different environment:")
    print("  source /scratch/hackathon-venv/bin/activate")
    print("\nThen install dependencies if needed:")
    print("  pip install torch torchvision numpy matplotlib pillow")
    print("=" * 70)
    sys.exit(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Try to import cv2, use alternative if not available
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Using PIL for resizing.")

# Add the cold_start_hackathon directory to path to import the model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cold_start_hackathon'))
from task import Net


class GradCAM:
    """
    Grad-CAM implementation for visualizing CNN attention maps.
    
    Grad-CAM uses the gradients of the target class flowing into the final
    convolutional layer to produce a localization map highlighting important
    regions in the image for predicting the class.
    """
    
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The neural network model
            target_layer: The target convolutional layer to generate CAM from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        # Use backward hook compatible with different PyTorch versions
        try:
            # Try new API first (PyTorch 1.8+)
            self.target_layer.register_full_backward_hook(self.save_gradient)
        except AttributeError:
            # Fall back to old API
            self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save the activation from forward pass."""
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save the gradient from backward pass."""
        # Handle different PyTorch versions
        if grad_output is not None and len(grad_output) > 0:
            self.gradients = grad_output[0]
        elif grad_input is not None and len(grad_input) > 0:
            self.gradients = grad_input[0]
    
    def generate_cam(self, input_image, class_idx=None):
        """
        Generate Class Activation Map (CAM) using Grad-CAM.
        
        Args:
            input_image: Input image tensor (batch_size, channels, height, width)
            class_idx: Target class index. If None, uses the predicted class.
        
        Returns:
            cam: Class activation map as numpy array
        """
        print("    → Setting model to evaluation mode...")
        self.model.eval()
        
        # Forward pass
        print("    → Running forward pass...")
        output = self.model(input_image)
        print(f"      ✓ Model output shape: {output.shape}")
        
        # For binary classification, we want to visualize the positive class
        # Backward pass - use the output directly (binary classification)
        print("    → Running backward pass to compute gradients...")
        self.model.zero_grad()
        
        # For binary classification, we visualize the positive class (output > 0)
        # Take the output value directly
        if output.dim() > 1:
            target = output[0, 0]  # Binary classification output
        else:
            target = output[0]
        
        print(f"      → Computing gradients for target: {target.item():.4f}")
        target.backward(retain_graph=True)
        print("      ✓ Gradients computed")
        
        # Get gradients and activations
        print("    → Extracting gradients and activations...")
        gradients = self.gradients[0]  # Shape: (channels, height, width)
        activations = self.activations[0]  # Shape: (channels, height, width)
        print(f"      ✓ Gradients shape: {gradients.shape}")
        print(f"      ✓ Activations shape: {activations.shape}")
        
        # Global average pooling of gradients
        print("    → Computing gradient weights (global average pooling)...")
        weights = torch.mean(gradients, dim=(1, 2))  # Shape: (channels,)
        print(f"      ✓ Computed {len(weights)} channel weights")
        
        # Weighted combination of activation maps
        print("    → Creating weighted activation map...")
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        print(f"      ✓ CAM shape: {cam.shape}")
        
        # Apply ReLU to get positive activations only
        print("    → Applying ReLU to get positive activations...")
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        print("    → Normalizing CAM to [0, 1]...")
        cam_min, cam_max = cam.min().item(), cam.max().item()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        print(f"      ✓ CAM normalized (min: {cam_min:.4f}, max: {cam_max:.4f})")
        
        return cam.detach().cpu().numpy()


def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load the trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint file (.pt)
        device: Device to load the model on
    
    Returns:
        model: Loaded model in evaluation mode
    """
    print("=" * 70)
    print("STEP 1: Loading Model")
    print("=" * 70)
    print(f"  Model path: {model_path}")
    print(f"  Device: {device}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    # Initialize model
    print("\n  → Initializing MobileNet V3 Large model architecture...")
    model = Net()
    print("  ✓ Model architecture initialized")
    
    # Load checkpoint
    print(f"\n  → Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    print("  ✓ Checkpoint loaded")
    
    # Handle different checkpoint formats
    print("  → Loading model weights...")
    state_dict_to_load = None
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict_to_load = checkpoint['model_state_dict']
            print("  ✓ Found 'model_state_dict'")
        elif 'state_dict' in checkpoint:
            state_dict_to_load = checkpoint['state_dict']
            print("  ✓ Found 'state_dict'")
        else:
            state_dict_to_load = checkpoint
            print("  ✓ Using dict root")
    else:
        state_dict_to_load = checkpoint
        print("  ✓ Using state dict directly")
    
    # Try loading with strict=False to handle architecture mismatches
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict_to_load, strict=False)
        if missing_keys:
            print(f"  ⚠ Warning: {len(missing_keys)} missing keys (architecture mismatch)")
            if len(missing_keys) < 10:
                print(f"    Missing: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"  ⚠ Warning: {len(unexpected_keys)} unexpected keys (architecture mismatch)")
            if len(unexpected_keys) < 10:
                print(f"    Unexpected: {unexpected_keys[:5]}...")
        print("  ✓ Model weights loaded (with architecture adaptation)")
    except Exception as e:
        print(f"  ✗ Error loading weights: {e}")
        print("  → Attempting partial load...")
        # Try to load only matching keys
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict_to_load.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"  ✓ Partially loaded {len(pretrained_dict)} matching layers")
    
    print(f"\n  → Moving model to {device}...")
    model.to(device)
    model.eval()
    print("  ✓ Model moved to device and set to evaluation mode")
    
    print("\n" + "=" * 70)
    print("✓ Model loaded successfully!")
    print("=" * 70 + "\n")
    return model


def preprocess_image(image_path, image_size=128):
    """
    Preprocess image for model input.
    
    Args:
        image_path: Path to the image file
        image_size: Target image size (default: 128)
    
    Returns:
        image_tensor: Preprocessed image tensor
        original_image: Original PIL image
    """
    print("  → Loading image...")
    # Load image
    if isinstance(image_path, str):
        original_image = Image.open(image_path).convert('L')  # Convert to grayscale
        print(f"    ✓ Image loaded: {original_image.size[0]}x{original_image.size[1]} pixels")
    else:
        original_image = image_path
        print(f"    ✓ Using provided image: {original_image.size[0]}x{original_image.size[1]} pixels")
    
    # Resize to match model input size
    print(f"  → Resizing to {image_size}x{image_size}...")
    original_image = original_image.resize((image_size, image_size), Image.LANCZOS)
    print("    ✓ Image resized")
    
    # Convert to numpy array and normalize to [0, 1]
    print("  → Converting to tensor and normalizing...")
    img_array = np.array(original_image, dtype=np.float32) / 255.0
    
    # Convert to tensor and add batch and channel dimensions
    image_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    print(f"    ✓ Image tensor shape: {image_tensor.shape}")
    
    return image_tensor, original_image


def apply_colormap(cam, colormap_name='jet'):
    """
    Apply colormap to CAM for visualization.
    
    Args:
        cam: Class activation map (numpy array)
        colormap_name: Name of the colormap (default: 'jet')
    
    Returns:
        colored_cam: Colored CAM
    """
    # Normalize CAM to 0-255
    cam_uint8 = (cam * 255).astype(np.uint8)
    
    # Apply colormap
    if colormap_name == 'hot':
        colored_cam = plt.cm.hot(cam)[:, :, :3]  # Remove alpha channel
    elif colormap_name == 'jet':
        colored_cam = plt.cm.jet(cam)[:, :, :3]
    else:
        colored_cam = plt.cm.get_cmap(colormap_name)(cam)[:, :, :3]
    
    return (colored_cam * 255).astype(np.uint8)


def visualize_explainability(model, image_path, output_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Generate and save explainability visualization.
    
    Args:
        model: The trained model
        image_path: Path to input image
        output_path: Path to save the visualization (optional)
        device: Device to run inference on
    """
    print("=" * 70)
    print("STEP 2: Processing Image")
    print("=" * 70)
    print(f"  Image path: {image_path}")
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path)
    print(f"\n  → Moving image tensor to {device}...")
    image_tensor = image_tensor.to(device)
    print("    ✓ Image tensor moved to device")
    
    # Get model prediction
    print("\n" + "=" * 70)
    print("STEP 3: Model Prediction")
    print("=" * 70)
    print("  → Running inference...")
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        # Handle different output shapes
        if output.dim() > 1:
            if output.size(1) == 1:
                logit = output[0, 0].item()
            else:
                logit = output[0].item()
        else:
            logit = output[0].item() if output.numel() > 1 else output.item()
        
        prediction = torch.sigmoid(torch.tensor(logit)).item()
        predicted_class = "Positive" if prediction > 0.5 else "Negative"
        confidence = prediction if prediction > 0.5 else 1 - prediction
    
    print(f"  ✓ Logit value: {logit:.4f}")
    print(f"  ✓ Prediction probability: {prediction:.4f}")
    print(f"  ✓ Predicted class: {predicted_class}")
    print(f"  ✓ Confidence: {confidence:.1%}")
    
    # Find the last convolutional layer in the features
    # MobileNet V3 features are in model.model.features
    print("\n" + "=" * 70)
    print("STEP 4: Setting Up Grad-CAM")
    print("=" * 70)
    print("  → Searching for target convolutional layer...")
    target_layer = None
    layer_count = 0
    for module in reversed(list(model.model.features.modules())):
        if isinstance(module, nn.Conv2d):
            target_layer = module
            layer_count += 1
            if layer_count == 1:  # Get the last one
                break
    
    if target_layer is None:
        raise ValueError("Could not find a suitable convolutional layer for Grad-CAM")
    
    print(f"  ✓ Found target layer: {type(target_layer).__name__}")
    print(f"    Layer details: {target_layer}")
    
    # Initialize Grad-CAM
    print("  → Initializing Grad-CAM...")
    grad_cam = GradCAM(model, target_layer)
    print("    ✓ Grad-CAM initialized with hooks registered")
    
    # Generate CAM
    print("\n" + "=" * 70)
    print("STEP 5: Generating Grad-CAM Heatmap")
    print("=" * 70)
    cam = grad_cam.generate_cam(image_tensor, class_idx=None)
    print(f"  ✓ CAM generated successfully!")
    print(f"    CAM shape: {cam.shape}")
    print(f"    CAM value range: [{cam.min():.4f}, {cam.max():.4f}]")
    
    # Resize CAM to match original image size
    print("\n" + "=" * 70)
    print("STEP 6: Creating Visualization")
    print("=" * 70)
    print(f"  → Resizing CAM from {cam.shape} to ({original_image.height}, {original_image.width})...")
    if HAS_CV2:
        cam_resized = cv2.resize(cam, (original_image.width, original_image.height))
        print("    ✓ Resized using OpenCV")
    else:
        # Use PIL for resizing if cv2 not available
        from PIL import Image as PILImage
        cam_pil = PILImage.fromarray((cam * 255).astype(np.uint8))
        cam_pil = cam_pil.resize((original_image.width, original_image.height), PILImage.LANCZOS)
        cam_resized = np.array(cam_pil).astype(np.float32) / 255.0
        print("    ✓ Resized using PIL")
    
    # Convert original image to RGB for overlay
    print("  → Converting image to RGB...")
    original_rgb = np.array(original_image.convert('RGB'))
    print("    ✓ Image converted")
    
    # Apply colormap to CAM
    print("  → Applying 'jet' colormap to heatmap...")
    heatmap = apply_colormap(cam_resized, colormap_name='jet')
    print("    ✓ Colormap applied")
    
    # Overlay heatmap on original image
    print("  → Creating overlay (60% original + 40% heatmap)...")
    if HAS_CV2:
        overlay = cv2.addWeighted(original_rgb, 0.6, heatmap, 0.4, 0)
        print("    ✓ Overlay created using OpenCV")
    else:
        # Manual blending if cv2 not available
        overlay = (0.6 * original_rgb.astype(np.float32) + 0.4 * heatmap.astype(np.float32)).astype(np.uint8)
        print("    ✓ Overlay created using manual blending")
    
    # Create visualization
    print("  → Creating matplotlib figure with 3 subplots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original X-ray Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap\n(Hotter = More Important)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\nPrediction: {predicted_class} ({confidence:.1%})', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    print("    ✓ Figure created")
    
    # Save or show
    if output_path:
        print(f"\n  → Saving visualization to {output_path}...")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    ✓ Visualization saved!")
    else:
        # Generate default output path
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_explainability.png"
        print(f"\n  → Saving visualization to {output_path}...")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"    ✓ Visualization saved!")
    
    plt.close()
    print("\n" + "=" * 70)
    
    return output_path


def main():
    """Main function to run explainability analysis."""
    parser = argparse.ArgumentParser(
        description='Generate explainability visualization for X-ray classification model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default model (best AUROC from models/ directory)
    python explainability.py --image_path data/xray.png
    
    # Specify model path explicitly
    python explainability.py --model_path models/job1033_rollv2-003_round9_auroc7637.pt --image_path data/xray.png
    
    # Specify custom output path
    python explainability.py --model_path models/job123_model.pt --image_path data/xray.png --output_path result.png
        """
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to the trained model checkpoint (.pt file). Default: looks for models in ./models/ directory'
    )
    
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='Path to the input X-ray image'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save the output visualization (default: <image_name>_explainability.png)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("EXPLAINABILITY ANALYSIS FOR X-RAY CLASSIFICATION")
    print("=" * 70)
    print(f"Starting explainability analysis...")
    print(f"Device: {args.device}")
    print("=" * 70 + "\n")
    
    # Handle default model path
    if args.model_path is None:
        print("No model path specified, searching for default model...")
        # Look for models in the models directory
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        if os.path.exists(models_dir):
            # Find the best model (highest AUROC) or most recent
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
            if model_files:
                # Sort by AUROC if available in filename, otherwise use most recent
                try:
                    # Try to extract AUROC from filename (format: jobXXX_*_roundY_aurocZZZZ.pt)
                    def get_auroc(filename):
                        parts = filename.split('_auroc')
                        if len(parts) > 1:
                            auroc_str = parts[1].replace('.pt', '')
                            try:
                                return int(auroc_str)
                            except:
                                return 0
                        return 0
                    
                    model_files.sort(key=get_auroc, reverse=True)
                    args.model_path = os.path.join(models_dir, model_files[0])
                    print(f"Using default model: {args.model_path}")
                except:
                    # Fallback to most recent file
                    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
                    args.model_path = os.path.join(models_dir, model_files[0])
                    print(f"Using most recent model: {args.model_path}")
            else:
                print("Error: No model files found in ./models/ directory")
                print("Please specify --model_path explicitly")
                sys.exit(1)
        else:
            print("Error: --model_path is required (models directory not found)")
            sys.exit(1)
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # Load model
    model = load_model(args.model_path, device=args.device)
    
    # Generate visualization
    try:
        output_path = visualize_explainability(
            model=model,
            image_path=args.image_path,
            output_path=args.output_path,
            device=args.device
        )
        print(f"\n✓ Successfully generated explainability visualization!")
        print(f"  Output saved to: {output_path}")
    except Exception as e:
        print(f"\n✗ Error generating visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

