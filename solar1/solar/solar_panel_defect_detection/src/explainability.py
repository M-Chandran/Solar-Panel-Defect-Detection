import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Any, Optional

# Try to import captum, but make it optional
try:
    from captum.attr import IntegratedGradients, GuidedGradCam, LayerGradCam
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    logging.warning("Captum not available. Some explainability features will be limited.")
    IntegratedGradients = None
    GuidedGradCam = None
    LayerGradCam = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplainableAIDefectDetector:
    """
    Explainable AI module for defect detection using multiple attribution methods.
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Initialize attribution methods (if captum is available)
        if CAPTUM_AVAILABLE:
            try:
                self.integrated_gradients = IntegratedGradients(self.model)
                self.guided_gradcam = GuidedGradCam(self.model, self.model.layer4)  # Using last conv layer
                self.layer_gradcam = LayerGradCam(self.model, self.model.layer4)
            except Exception as e:
                logger.warning(f"Failed to initialize captum attribution methods: {e}")
                self.integrated_gradients = None
                self.guided_gradcam = None
                self.layer_gradcam = None
        else:
            self.integrated_gradients = None
            self.guided_gradcam = None
            self.layer_gradcam = None

        logger.info("ExplainableAI module initialized")

    def generate_explanations(self, image, target_class=None) -> Dict[str, Any]:
        """
        Generate comprehensive explanations for defect detection.

        Args:
            image: Input image tensor [1, 3, H, W]
            target_class: Target class for explanation (None for highest scoring)

        Returns:
            explanations: Dict containing various explanation maps and metadata
        """
        image = image.to(self.device)

        # Get model prediction
        with torch.no_grad():
            output = self.model(image)
            if target_class is None:
                target_class = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, target_class].item()

        # Generate different attribution maps
        explanations = {
            'prediction': {
                'class': target_class,
                'confidence': confidence
            },
            'attribution_maps': {},
            'feature_importance': {},
            'decision_factors': []
        }

        # 1. Integrated Gradients
        ig_attr = self._compute_integrated_gradients(image, target_class)
        explanations['attribution_maps']['integrated_gradients'] = ig_attr

        # 2. Guided Grad-CAM
        guided_gc_attr = self._compute_guided_gradcam(image, target_class)
        explanations['attribution_maps']['guided_gradcam'] = guided_gc_attr

        # 3. Layer Grad-CAM
        layer_gc_attr = self._compute_layer_gradcam(image, target_class)
        explanations['attribution_maps']['layer_gradcam'] = layer_gc_attr

        # 4. Feature importance analysis
        feature_imp = self._analyze_feature_importance(image)
        explanations['feature_importance'] = feature_imp

        # 5. Generate natural language explanations
        nl_explanation = self._generate_natural_language_explanation(
            explanations, image.shape
        )
        explanations['natural_language'] = nl_explanation

        return explanations

    def _compute_integrated_gradients(self, image, target_class):
        """Compute Integrated Gradients attribution."""
        try:
            # Create baseline (black image)
            baseline = torch.zeros_like(image)

            # Compute attributions
            attributions = self.integrated_gradients.attribute(
                image, baseline, target=target_class, n_steps=50
            )

            # Convert to numpy and normalize
            attr = attributions.squeeze().cpu().detach().numpy()
            attr = np.transpose(attr, (1, 2, 0))  # [H, W, 3]

            # Convert to grayscale attribution map
            attr_gray = np.mean(np.abs(attr), axis=2)  # [H, W]
            attr_gray = (attr_gray - attr_gray.min()) / (attr_gray.max() - attr_gray.min() + 1e-8)

            return attr_gray

        except Exception as e:
            logger.warning(f"Integrated Gradients failed: {e}")
            return np.zeros((image.shape[2], image.shape[3]))

    def _compute_guided_gradcam(self, image, target_class):
        """Compute Guided Grad-CAM attribution."""
        try:
            attributions = self.guided_gradcam.attribute(image, target=target_class)

            # Convert to numpy and process
            attr = attributions.squeeze().cpu().detach().numpy()
            attr = np.transpose(attr, (1, 2, 0))  # [H, W, 3]

            # Convert to grayscale
            attr_gray = np.mean(np.abs(attr), axis=2)
            attr_gray = (attr_gray - attr_gray.min()) / (attr_gray.max() - attr_gray.min() + 1e-8)

            return attr_gray

        except Exception as e:
            logger.warning(f"Guided Grad-CAM failed: {e}")
            return np.zeros((image.shape[2], image.shape[3]))

    def _compute_layer_gradcam(self, image, target_class):
        """Compute Layer Grad-CAM attribution."""
        try:
            attributions = self.layer_gradcam.attribute(image, target=target_class)

            # Convert to numpy
            attr = attributions.squeeze().cpu().detach().numpy()  # [H, W]

            # Normalize
            attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)

            return attr

        except Exception as e:
            logger.warning(f"Layer Grad-CAM failed: {e}")
            return np.zeros((image.shape[2], image.shape[3]))

    def _analyze_feature_importance(self, image) -> Dict[str, Any]:
        """Analyze which features contributed most to the decision."""
        # Extract intermediate features
        feature_importance = {}

        # Hook to capture intermediate features
        intermediate_features = {}

        def hook_fn(module, input, output, name):
            intermediate_features[name] = output.detach()

        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and 'layer4' in name:
                hooks.append(module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                ))

        # Forward pass
        with torch.no_grad():
            _ = self.model(image)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Analyze feature statistics
        if intermediate_features:
            for name, features in intermediate_features.items():
                # Compute feature statistics
                feature_map = features.squeeze().cpu().numpy()  # [C, H, W]
                mean_activation = np.mean(feature_map, axis=(1, 2))
                std_activation = np.std(feature_map, axis=(1, 2))
                max_activation = np.max(feature_map, axis=(1, 2))

                feature_importance[name] = {
                    'mean_activation': mean_activation.tolist(),
                    'std_activation': std_activation.tolist(),
                    'max_activation': max_activation.tolist(),
                    'num_channels': feature_map.shape[0]
                }

        return feature_importance

    def _generate_natural_language_explanation(self, explanations, image_shape) -> str:
        """Generate natural language explanation of the model's decision."""
        prediction = explanations['prediction']
        confidence = prediction['confidence']

        # Analyze attribution maps
        ig_map = explanations['attribution_maps'].get('integrated_gradients', np.zeros(image_shape[2:]))
        gc_map = explanations['attribution_maps'].get('guided_gradcam', np.zeros(image_shape[2:]))

        # Compute statistics
        ig_coverage = np.sum(ig_map > 0.5) / ig_map.size
        gc_coverage = np.sum(gc_map > 0.5) / gc_map.size

        # Generate explanation
        explanation_parts = []

        if confidence > 0.8:
            explanation_parts.append(f"The model is highly confident ({confidence:.2f}) in its prediction.")
        elif confidence > 0.6:
            explanation_parts.append(f"The model is moderately confident ({confidence:.2f}) in its prediction.")
        else:
            explanation_parts.append(f"The model has low confidence ({confidence:.2f}) in its prediction.")

        # Analyze spatial focus
        if ig_coverage > 0.3:
            explanation_parts.append("The model focused on a significant portion of the image.")
        elif ig_coverage > 0.1:
            explanation_parts.append("The model focused on specific regions of the image.")
        else:
            explanation_parts.append("The model focused on very localized areas of the image.")

        # Feature analysis
        feature_imp = explanations.get('feature_importance', {})
        if feature_imp:
            total_channels = sum(info['num_channels'] for info in feature_imp.values())
            active_channels = sum(np.sum(np.array(info['mean_activation']) > 0.1)
                                for info in feature_imp.values())
            explanation_parts.append(f"The model activated {active_channels}/{total_channels} feature channels.")

        explanation = " ".join(explanation_parts)
        return explanation

    def visualize_explanations(self, image, explanations, save_path=None):
        """
        Create visualization of explanations.

        Args:
            image: Original image [H, W, 3]
            explanations: Explanations dict from generate_explanations
            save_path: Path to save visualization (optional)
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Explainable AI: Defect Detection Analysis', fontsize=16)

        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Attribution maps
        attr_maps = explanations['attribution_maps']

        # Integrated Gradients
        if 'integrated_gradients' in attr_maps:
            axes[0, 1].imshow(attr_maps['integrated_gradients'], cmap='hot')
            axes[0, 1].set_title('Integrated Gradients')
            axes[0, 1].axis('off')

        # Guided Grad-CAM
        if 'guided_gradcam' in attr_maps:
            axes[0, 2].imshow(attr_maps['guided_gradcam'], cmap='hot')
            axes[0, 2].set_title('Guided Grad-CAM')
            axes[0, 2].axis('off')

        # Layer Grad-CAM
        if 'layer_gradcam' in attr_maps:
            axes[1, 0].imshow(attr_maps['layer_gradcam'], cmap='hot')
            axes[1, 0].set_title('Layer Grad-CAM')
            axes[1, 0].axis('off')

        # Feature importance histogram
        feature_imp = explanations.get('feature_importance', {})
        if feature_imp:
            all_means = []
            for info in feature_imp.values():
                all_means.extend(info['mean_activation'])
            axes[1, 1].hist(all_means, bins=20, alpha=0.7)
            axes[1, 1].set_title('Feature Activation Distribution')
            axes[1, 1].set_xlabel('Activation Value')
            axes[1, 1].set_ylabel('Frequency')

        # Text explanation
        nl_explanation = explanations.get('natural_language', '')
        axes[1, 2].text(0.1, 0.5, nl_explanation, wrap=True,
                       fontsize=10, verticalalignment='center')
        axes[1, 2].set_title('Natural Language Explanation')
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Explanation visualization saved to {save_path}")

        return fig

class UncertaintyEstimator:
    """
    Estimate uncertainty in model predictions using Monte Carlo dropout.
    """
    def __init__(self, model, num_samples=10):
        self.model = model
        self.num_samples = num_samples

        # Enable dropout in evaluation mode
        self._enable_dropout()

    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def estimate_uncertainty(self, image):
        """
        Estimate prediction uncertainty using MC Dropout.

        Args:
            image: Input image tensor [1, 3, H, W]

        Returns:
            mean_prediction: Mean prediction across samples
            uncertainty: Prediction uncertainty (variance)
        """
        predictions = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(image)
                predictions.append(torch.softmax(output, dim=1).cpu().numpy())

        predictions = np.array(predictions)  # [num_samples, 1, num_classes]
        predictions = predictions.squeeze(1)  # [num_samples, num_classes]

        mean_prediction = np.mean(predictions, axis=0)
        uncertainty = np.var(predictions, axis=0)

        return mean_prediction, uncertainty

if __name__ == "__main__":
    # Test the explainability module
    print("Testing ExplainableAI module...")

    # Create mock model
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Conv2d(3, 64, 3, padding=1)
            self.layer2 = nn.Conv2d(64, 128, 3, padding=1)
            self.layer3 = nn.Conv2d(128, 256, 3, padding=1)
            self.layer4 = nn.Conv2d(256, 512, 3, padding=1)
            self.classifier = nn.Linear(512, 4)  # 4 defect classes

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.layer2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.layer3(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.layer4(x))
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    model = MockModel()
    explainer = ExplainableAIDefectDetector(model)

    # Test with dummy image
    test_image = torch.randn(1, 3, 224, 224)
    explanations = explainer.generate_explanations(test_image)

    print(f"Generated explanations with keys: {list(explanations.keys())}")
    print(f"Prediction: {explanations['prediction']}")
    print(f"Natural language explanation: {explanations['natural_language']}")

    print("ExplainableAI test completed!")
