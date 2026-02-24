"""
Test script to process solar.jpg with the MoCo-based defect detection framework.
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.moco import MoCo
from src.data_preprocessing import preprocess_image
from src.classification import DefectClassifier

def load_and_preprocess_image(image_path):
    """Load and preprocess the solar panel image."""
    print(f"Loading image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Original image shape: {image.shape}")
    
    # Preprocess for model input
    processed = preprocess_image(image)
    
    return image, processed

def extract_features(model, image_tensor):
    """Extract features using MoCo model."""
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Move to same device as model
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # Extract features
        features = model.extract_features(image_tensor)
        
    return features.cpu().numpy()

def analyze_image(image_path):
    """Complete analysis of solar panel image."""
    print("="*60)
    print("SOLAR PANEL DEFECT DETECTION - MoCo + KNN Framework")
    print("="*60)
    
    # Initialize model
    print("\n[1] Initializing MoCo model with SE-ResNet50...")
    model = MoCo(
        dim=128,
        K=4096,
        m=0.999,
        T=0.2,
        arch='se_resnet50',
        pretrained=False  # Using random weights for demo (should be trained)
    )
    
    # Load checkpoint if available
    checkpoint_path = 'checkpoint.pth.tar'
    if os.path.exists(checkpoint_path):
        print(f"   Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print("   ✓ Checkpoint loaded")
    else:
        print("   ⚠ No checkpoint found - using untrained model (for demo only)")
    
    model.eval()
    
    # Load and preprocess image
    print("\n[2] Loading and preprocessing image...")
    original_img, processed_img = load_and_preprocess_image(image_path)
    print(f"   ✓ Image preprocessed: {processed_img.shape}")
    
    # Extract features
    print("\n[3] Extracting features with MoCo...")
    features = extract_features(model, processed_img)
    print(f"   ✓ Feature vector shape: {features.shape}")
    print(f"   ✓ Feature statistics:")
    print(f"     - Mean: {features.mean():.4f}")
    print(f"     - Std: {features.std():.4f}")
    print(f"     - Min: {features.min():.4f}")
    print(f"     - Max: {features.max():.4f}")
    
    # Simulate classification (would need trained KNN for real predictions)
    print("\n[4] Classification Results (Demo with untrained model):")
    print("   " + "-"*50)
    
    # For demonstration, show what the output would look like
    # In practice, you would use a trained KNN classifier here
    
    # Simulate confidence based on feature magnitude
    feature_norm = np.linalg.norm(features)
    simulated_confidence = min(0.95, 0.5 + feature_norm / 100)
    
    # Random prediction for demo (would be from trained KNN)
    np.random.seed(42)
    prediction = "NORMAL" if np.random.random() > 0.3 else "DEFECTIVE"
    
    print(f"   Predicted Class: {prediction}")
    print(f"   Confidence: {simulated_confidence:.2%}")
    print(f"   Feature Norm: {feature_norm:.4f}")
    
    # Show additional analysis
    print("\n[5] Detailed Analysis:")
    print("   " + "-"*50)
    
    # Analyze different regions (simulated)
    h, w = original_img.shape[:2]
    regions = {
        "Top-Left": original_img[0:h//2, 0:w//2],
        "Top-Right": original_img[0:h//2, w//2:w],
        "Bottom-Left": original_img[h//2:h, 0:w//2],
        "Bottom-Right": original_img[h//2:h, w//2:w],
        "Center": original_img[h//4:3*h//4, w//4:3*w//4]
    }
    
    print("   Regional Analysis:")
    for region_name, region_img in regions.items():
        # Calculate simple statistics for each region
        brightness = np.mean(region_img)
        contrast = np.std(region_img)
        print(f"     {region_name:12s}: Brightness={brightness:6.1f}, Contrast={contrast:6.1f}")
    
    # Generate visualization
    print("\n[6] Generating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Preprocessed image (convert tensor back to display)
    img_display = processed_img.permute(1, 2, 0).numpy()
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    axes[0, 1].imshow(img_display)
    axes[0, 1].set_title('Preprocessed (224x224)')
    axes[0, 1].axis('off')
    
    # Feature visualization (as heatmap)
    feature_map = features.reshape(1, -1)
    axes[0, 2].imshow(feature_map, cmap='viridis', aspect='auto')
    axes[0, 2].set_title(f'Feature Vector ({features.shape[1]}D)')
    axes[0, 2].set_xlabel('Feature Dimension')
    
    # Regional analysis visualization
    region_names = list(regions.keys())
    brightness_vals = [np.mean(regions[r]) for r in region_names]
    contrast_vals = [np.std(regions[r]) for r in region_names]
    
    x_pos = np.arange(len(region_names))
    axes[1, 0].bar(x_pos, brightness_vals, color='skyblue')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(region_names, rotation=45, ha='right')
    axes[1, 0].set_title('Regional Brightness')
    axes[1, 0].set_ylabel('Mean Brightness')
    
    axes[1, 1].bar(x_pos, contrast_vals, color='coral')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(region_names, rotation=45, ha='right')
    axes[1, 1].set_title('Regional Contrast')
    axes[1, 1].set_ylabel('Std Dev')
    
    # Prediction result
    axes[1, 2].text(0.5, 0.7, f'Prediction: {prediction}', 
                    fontsize=20, ha='center', weight='bold',
                    color='green' if prediction == 'NORMAL' else 'red')
    axes[1, 2].text(0.5, 0.5, f'Confidence: {simulated_confidence:.2%}', 
                    fontsize=16, ha='center')
    axes[1, 2].text(0.5, 0.3, f'Feature Norm: {feature_norm:.4f}', 
                    fontsize=14, ha='center')
    axes[1, 2].text(0.5, 0.1, 'Note: Demo with untrained model', 
                    fontsize=10, ha='center', style='italic', color='gray')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Classification Result')
    
    plt.tight_layout()
    
    # Save result
    output_path = 'solar_test_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Result saved: {output_path}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nTo get real predictions:")
    print("1. Train the model: python src/train_ssl.py --data data/ --epochs 200")
    print("2. Train KNN classifier on labeled data")
    print("3. Run this script again with trained weights")
    
    return {
        'prediction': prediction,
        'confidence': simulated_confidence,
        'features': features,
        'feature_norm': feature_norm
    }

if __name__ == "__main__":
    # Test with solar.jpg
    image_path = "solar.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found!")
        print("Please ensure solar.jpg is in the current directory.")
        sys.exit(1)
    
    try:
        results = analyze_image(image_path)
        print("\nResults summary:", results)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
