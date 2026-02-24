"""
Test script for MoCo-based Self-Supervised Learning Framework.
Verifies all components work correctly before training.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.moco import MoCo, SEResNet50, InfoNCELoss
from src.data_preprocessing import SolarPanelDataset, get_data_loaders, preprocess_image
from src.classification import DefectClassifier

def test_se_resnet50():
    """Test SE-ResNet50 encoder."""
    print("\n" + "="*50)
    print("Testing SE-ResNet50 Encoder...")
    print("="*50)
    
    try:
        model = SEResNet50(pretrained=False, num_classes=0)
        x = torch.randn(2, 3, 224, 224)
        
        # Test forward pass through backbone
        features = model(x)
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {features.shape}")
        print("✓ SE-ResNet50 test passed!")
        return True
    except Exception as e:
        print(f"✗ SE-ResNet50 test failed: {e}")
        return False

def test_moco_model():
    """Test MoCo model with SE-ResNet50."""
    print("\n" + "="*50)
    print("Testing MoCo Model...")
    print("="*50)
    
    try:
        # Create MoCo model with paper's hyperparameters
        model = MoCo(
            dim=128,      # Feature dimension
            K=4096,       # Queue size
            m=0.999,      # Momentum
            T=0.2,        # Temperature
            arch='se_resnet50',
            pretrained=False
        )
        
        # Test forward pass
        im_q = torch.randn(4, 3, 224, 224)
        im_k = torch.randn(4, 3, 224, 224)
        
        logits, labels = model(im_q, im_k)
        
        print(f"✓ Query images shape: {im_q.shape}")
        print(f"✓ Key images shape: {im_k.shape}")
        print(f"✓ Logits shape: {logits.shape}")
        print(f"✓ Labels shape: {labels.shape}")
        print(f"✓ Queue size: {model.K}")
        print(f"✓ Temperature: {model.T}")
        print(f"✓ Momentum: {model.m}")
        
        # Test feature extraction for KNN
        features = model.extract_features(im_q)
        print(f"✓ Extracted features shape: {features.shape}")
        
        print("✓ MoCo model test passed!")
        return True
        
    except Exception as e:
        print(f"✗ MoCo model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_augmentation():
    """Test ten-crop data augmentation."""
    print("\n" + "="*50)
    print("Testing Ten-Crop Data Augmentation...")
    print("="*50)
    
    try:
        # Create dummy dataset
        data_dir = "data"
        if not os.path.exists(data_dir):
            print(f"⚠ Data directory {data_dir} not found. Creating dummy test...")
            # Create test with synthetic data
            dataset = SolarPanelDataset(data_dir, is_train=True)
            
            # Test ten-crop augmentation directly
            dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            crops = dataset._ten_crop_augmentation(dummy_image, crop_size=224, scale_factor=1.2)
            
            print(f"✓ Generated {len(crops)} crops (5 crops + 5 mirrored)")
            print(f"✓ Each crop shape: {crops[0].shape}")
            
            # Test full augmentation pipeline
            aug1 = dataset._augment_image(dummy_image)
            aug2 = dataset._augment_image(dummy_image)
            
            print(f"✓ Augmented view 1 shape: {aug1.shape}")
            print(f"✓ Augmented view 2 shape: {aug2.shape}")
            print(f"✓ Views are different: {not torch.allclose(aug1, aug2)}")
            
        else:
            train_loader, val_loader = get_data_loaders(data_dir, batch_size=4)
            print(f"✓ Train loader created with {len(train_loader)} batches")
            print(f"✓ Val loader created with {len(val_loader)} batches")
            
            # Test one batch
            batch = next(iter(train_loader))
            view1, view2 = batch
            print(f"✓ Batch view1 shape: {view1.shape}")
            print(f"✓ Batch view2 shape: {view2.shape}")
        
        print("✓ Data augmentation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Data augmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_knn_classifier():
    """Test KNN classifier with MoCo features."""
    print("\n" + "="*50)
    print("Testing KNN Classifier...")
    print("="*50)
    
    try:
        # Create MoCo model
        model = MoCo(dim=128, K=4096, m=0.999, T=0.2, arch='se_resnet50', pretrained=False)
        model.eval()
        
        # Create classifier
        classifier = DefectClassifier(n_neighbors=5, defect_types=['normal', 'defective'])
        
        # Create dummy training data
        print("Generating dummy training data...")
        train_images = [np.random.rand(224, 224, 3) for _ in range(20)]
        train_labels = [0] * 10 + [1] * 10  # 10 normal, 10 defective
        
        training_data = {
            'images': train_images,
            'labels': train_labels
        }
        
        # Train classifier
        print("Training KNN classifier...")
        classifier.train(model, training_data)
        
        # Test prediction
        print("Testing predictions...")
        test_images = [np.random.rand(224, 224, 3) for _ in range(5)]
        predictions, confidences = classifier.predict(model, test_images)
        
        print(f"✓ Predictions: {predictions}")
        print(f"✓ Confidences: {[f'{c:.3f}' for c in confidences]}")
        
        # Test evaluation
        print("Testing evaluation...")
        test_data = {
            'images': test_images,
            'labels': [0, 1, 0, 1, 0]
        }
        
        metrics = classifier.evaluate(model, test_data)
        print(f"✓ Accuracy: {metrics['accuracy']:.3f}")
        
        print("✓ KNN classifier test passed!")
        return True
        
    except Exception as e:
        print(f"✗ KNN classifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_function():
    """Test InfoNCE loss function."""
    print("\n" + "="*50)
    print("Testing InfoNCE Loss...")
    print("="*50)
    
    try:
        # Create loss function with paper's temperature
        criterion = InfoNCELoss(temperature=0.2)
        
        # Create dummy logits and labels
        batch_size = 4
        queue_size = 4096
        logits = torch.randn(batch_size, 1 + queue_size)  # 1 positive + K negatives
        labels = torch.zeros(batch_size, dtype=torch.long)
        
        loss = criterion(logits, labels)
        
        print(f"✓ Logits shape: {logits.shape}")
        print(f"✓ Labels shape: {labels.shape}")
        print(f"✓ Loss value: {loss.item():.4f}")
        print(f"✓ Temperature: {criterion.temperature}")
        
        print("✓ InfoNCE loss test passed!")
        return True
        
    except Exception as e:
        print(f"✗ InfoNCE loss test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("MoCo-Based Self-Supervised Learning Framework - Test Suite")
    print("="*70)
    
    results = {
        'SE-ResNet50': test_se_resnet50(),
        'MoCo Model': test_moco_model(),
        'Data Augmentation': test_data_augmentation(),
        'KNN Classifier': test_knn_classifier(),
        'InfoNCE Loss': test_loss_function()
    }
    
    print("\n" + "="*70)
    print("Test Results Summary")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("="*70)
    if all_passed:
        print("🎉 All tests passed! Framework is ready for training.")
        print("\nNext steps:")
        print("1. Prepare your dataset in the 'data/' directory")
        print("2. Run: python src/train_ssl.py --data data/")
        print("3. After training, use the trained model with KNN classifier")
    else:
        print("⚠ Some tests failed. Please review the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
