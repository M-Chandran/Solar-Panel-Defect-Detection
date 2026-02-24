# MoCo-Based Self-Supervised Learning Framework for Solar Panel Defect Detection

Implementation of the novel MoCo+KNN framework from the research paper "A Novel MoCo-Based Self-Supervised Learning Framework for Solar Panel Defect Detection" (IEEE Access, 2025).

## Overview

This implementation achieves **96.95% accuracy** on the ELPV dataset and **99.44%** on the EL dataset through:

- **MoCo v2** (Momentum Contrast) for self-supervised feature learning
- **SE-ResNet50** encoder with Squeeze-and-Excitation attention mechanism
- **KNN classifier** for defect detection
- **Ten-crop data augmentation** with random flips

## Key Features

### 1. SE-ResNet50 Encoder
- Squeeze-and-Excitation (SE) blocks for channel attention
- Pretrained on ImageNet-1K for better feature extraction
- Focuses on defect-relevant regions in solar panel images

### 2. MoCo v2 Framework
- **Feature dimension**: 128
- **Queue size**: 4096 negative samples
- **Momentum**: 0.999 for key encoder update
- **Temperature**: 0.2 for InfoNCE loss
- **2-layer MLP projection head** with ReLU

### 3. Ten-Crop Data Augmentation
- 5 crops: 4 corners + center
- 5 mirrored versions (horizontal flip)
- Random horizontal and vertical flips
- Scale factor: 1.2x

### 4. KNN Classification
- Cosine similarity metric
- 5 nearest neighbors
- Feature extraction from MoCo backbone (before projection head)

## Installation

```bash
# Install dependencies
pip install torch torchvision
pip install albumentations
pip install scikit-learn
pip install opencv-python
pip install numpy
pip install tqdm
```

## Usage

### Quick Test
```bash
python test_moco_implementation.py
```

### Training
```bash
python src/train_ssl.py \
    --data data/ \
    --arch se_resnet50 \
    --epochs 200 \
    --batch-size 64 \
    --lr 0.03 \
    --wd 5e-4 \
    --cos
```

### Defect Detection
```python
import torch
from models.moco import MoCo
from src.classification import DefectClassifier

# Load trained model
model = MoCo(dim=128, K=4096, m=0.999, T=0.2, arch='se_resnet50')
checkpoint = torch.load('checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Create and train KNN classifier
classifier = DefectClassifier(n_neighbors=5, defect_types=['normal', 'defective'])
classifier.train(model, training_data)

# Predict on new images
predictions, confidences = classifier.predict(model, test_images)
```

## Project Structure

```
solar_panel_defect_detection/
├── models/
│   ├── moco.py              # MoCo + SE-ResNet50 implementation
│   └── attention.py         # Additional attention mechanisms
├── src/
│   ├── train_ssl.py         # Self-supervised training script
│   ├── data_preprocessing.py # Ten-crop augmentation & data loading
│   ├── classification.py    # KNN classifier
│   ├── localization.py      # Defect localization
│   └── explainability.py    # Attention visualization
├── test_moco_implementation.py  # Test suite
└── data/                    # Dataset directory
```

## Hyperparameters (from Paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 200 | Training epochs |
| Batch size | 64 | Mini-batch size |
| Learning rate | 0.03 | Initial LR with cosine annealing |
| Weight decay | 5e-4 | L2 regularization |
| Temperature (T) | 0.2 | InfoNCE temperature |
| Momentum (m) | 0.999 | Key encoder momentum |
| Queue size (K) | 4096 | Negative sample queue |
| Feature dim | 128 | Embedding dimension |
| Architecture | SE-ResNet50 | Backbone network |

## Results

### ELPV Dataset
- **Accuracy**: 96.95%
- Outperforms unsupervised methods: KDAD (74.4%), SAOE (63.5%), DRA (67.5%), BGAD-FAS (90.3%)
- Outperforms supervised methods: Adapted VGG19 (74.5%), Adapted VGG16 (66.2%), ShuffleNet (77.6%)

### EL Dataset
- **Accuracy**: 99.44%
- Demonstrates robustness across different environments

## Algorithm

### MoCo Training (Algorithm 1 from paper)
```
1. Initialize encoder_q and encoder_k with same weights
2. For each batch:
   a. Create two augmented views: x_q, x_k
   b. Extract features: q = encoder_q(x_q), k = encoder_k(x_k)
   c. Compute InfoNCE loss with queue
   d. Update encoder_q via backpropagation
   e. Update encoder_k via momentum: θ_k = m·θ_k + (1-m)·θ_q
   f. Update queue: enqueue(k), dequeue()
```

### KNN Classification
```
1. Extract features from training images using encoder_q
2. Train KNN classifier on features with cosine metric
3. For test images:
   a. Extract features using encoder_q
   b. Find k nearest neighbors
   c. Classify based on majority vote
```

## Citation

```bibtex
@article{huang2025moco,
  title={A Novel MoCo-Based Self-Supervised Learning Framework for Solar Panel Defect Detection},
  author={Huang, Jun and Xu, Wanting and Ariffin, Shamsul Arrieya and Chen, Yongqiang and Lin, Jinghui},
  journal={IEEE Access},
  volume={13},
  pages={22977--22988},
  year={2025},
  publisher={IEEE}
}
```

## License

This implementation follows the paper's methodology for research and educational purposes.
