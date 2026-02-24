# Solar Panel Defect Detection System

An autonomous AI system for detecting and localizing solar panel defects in electroluminescence (EL) or RGB images without requiring labeled data. This system uses self-supervised learning, attention mechanisms, and explainable AI to identify defects like cracks, hotspots, and cell damage.

## Features

- **Self-Supervised Learning**: MoCo v2-based representation learning without labels
- **Defect Localization**: Visual labeling of defective regions using patch analysis and heatmaps
- **Explainable AI**: Grad-CAM++ and attention maps for decision explanations
- **Web Application**: Flask backend with HTML/CSS/JS frontend for easy deployment
- **Real-Time Processing**: Support for image uploads and camera integration
- **Deployment Ready**: Dockerized with GPU acceleration and optimization

## Architecture

1. **Hybrid Detection Pipeline**: Confidence-weighted fusion of rule-based and deep learning methods
2. **Advanced Data Processing**: Multi-view augmentation, contrastive learning, and preprocessing
3. **Self-Supervised Learning**: MoCo v2 with attention mechanisms for robust representations
4. **Multi-Scale Localization**: Patch analysis, Grad-CAM++, attention maps, and ensemble heatmaps
5. **Intelligent Classification**: KNN, GNN, and ensemble methods with uncertainty estimation
6. **Comprehensive Explainability**: Integrated Gradients, SHAP, counterfactuals, and NL explanations
7. **Real-Time Systems**: Camera integration, video processing, and edge optimization
8. **Production Infrastructure**: Docker, monitoring, CI/CD, and scalable deployment

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd solar_panel_defect_detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) For GPU support, ensure CUDA is installed.

## Usage

### Training Self-Supervised Model

```bash
python src/train_ssl.py --data_path data/ --epochs 100
```

### Running Web Application

```bash
python web/backend/app.py
```

Open `http://localhost:5000` in your browser.

### API Endpoints

- `POST /upload`: Upload image for processing
- `GET /infer`: Run inference on uploaded image
- `GET /heatmap`: Get defect heatmap overlay
- `GET /result`: Get classification results and explanations

## Datasets

- ELPV Dataset: https://github.com/zae-bayern/elpv-dataset
- EL Dataset: https://github.com/zae-bayern/elpv-dataset (extended)

Place datasets in `data/` directory.

## Evaluation

Run evaluation script:
```bash
python tests/evaluate.py
```

Metrics: Accuracy ≥95%, Localization Precision, Inference Latency.

## Deployment

Build Docker image:
```bash
docker build -t solar-defect-detection .
docker run -p 5000:5000 solar-defect-detection
```

## Future Extensions

- Real-time camera defect tracking
- Continual learning for new defect types
- Edge-AI optimization for mobile devices
- Cross-industry defect detection (e.g., manufacturing)

## Contributing

Please read CONTRIBUTING.md for details on code style and contribution process.

## License

This project is licensed under the MIT License - see LICENSE.md for details.

## Acknowledgments

- MoCo v2: https://arxiv.org/abs/2003.04297
- Albumentations: https://albumentations.ai/
- Captum for Explainability: https://captum.ai/
