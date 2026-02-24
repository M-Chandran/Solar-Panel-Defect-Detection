import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import logging

from models.attention import DefectAttentionEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatchEmbedder(nn.Module):
    """
    Embeds image patches into feature space for anomaly detection.
    """
    def __init__(self, patch_size=16, embed_dim=256):
        super(PatchEmbedder, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Convolutional embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, 3, H, W]
        patches = self.patch_embed(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        patches = self.norm(patches)
        return patches

class PrototypeMemory(nn.Module):
    """
    Memory bank for storing healthy patch prototypes.
    """
    def __init__(self, num_prototypes=1000, feature_dim=256):
        super(PrototypeMemory, self).__init__()
        self.num_prototypes = num_prototypes
        self.feature_dim = feature_dim

        # Initialize prototypes randomly
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        self.prototypes.data = F.normalize(self.prototypes.data, dim=1)

    def update_prototypes(self, features, momentum=0.9):
        """
        Update prototypes using momentum update.
        features: [N, feature_dim]
        """
        with torch.no_grad():
            # Normalize features
            features = F.normalize(features, dim=1)

            # Simple prototype update (can be improved with clustering)
            batch_size = features.shape[0]
            if batch_size > self.num_prototypes:
                # Randomly select prototypes to update
                indices = torch.randperm(batch_size)[:self.num_prototypes]
                selected_features = features[indices]
            else:
                selected_features = features

            # Momentum update
            self.prototypes.data = momentum * self.prototypes.data + (1 - momentum) * selected_features.mean(dim=0)

            # Renormalize
            self.prototypes.data = F.normalize(self.prototypes.data, dim=1)

    def get_similarity_scores(self, features):
        """
        Compute cosine similarity between features and prototypes.
        features: [N, feature_dim]
        Returns: [N, num_prototypes]
        """
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        similarity = torch.matmul(features, prototypes.t())  # [N, num_prototypes]
        return similarity

class PatchAnomalyDetector(nn.Module):
    """
    Detects anomalies at patch level using prototype-based approach.
    """
    def __init__(self, patch_size=16, embed_dim=256, num_prototypes=1000):
        super(PatchAnomalyDetector, self).__init__()
        self.patch_embedder = PatchEmbedder(patch_size, embed_dim)
        self.prototype_memory = PrototypeMemory(num_prototypes, embed_dim)

    def forward(self, x):
        # x: [B, 3, H, W]
        batch_size = x.shape[0]

        # Extract patch features
        patch_features = self.patch_embedder(x)  # [B, num_patches, embed_dim]

        # Flatten batch and patches for prototype comparison
        patch_features_flat = patch_features.view(-1, patch_features.shape[-1])  # [B*num_patches, embed_dim]

        # Get similarity to healthy prototypes
        similarities = self.prototype_memory.get_similarity_scores(patch_features_flat)  # [B*num_patches, num_prototypes]

        # Anomaly score: distance to nearest prototype (lower similarity = higher anomaly)
        max_similarities, _ = torch.max(similarities, dim=1)  # [B*num_patches]
        anomaly_scores = 1 - max_similarities  # [B*num_patches]

        # Reshape back to spatial dimensions
        H_patches = x.shape[2] // self.patch_embedder.patch_size
        W_patches = x.shape[3] // self.patch_embedder.patch_size
        anomaly_map = anomaly_scores.view(batch_size, H_patches, W_patches)  # [B, H_patches, W_patches]

        return anomaly_map, patch_features

    def update_memory(self, healthy_features):
        """
        Update prototype memory with healthy patch features.
        healthy_features: [N, embed_dim]
        """
        self.prototype_memory.update_prototypes(healthy_features)

class GradCAMPlusPlus:
    """
    Grad-CAM++ for generating attention heatmaps.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to capture gradients and activations
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM++ heatmap.
        input_tensor: [1, C, H, W]
        class_idx: target class index (None for highest scoring class)
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        # Backward pass
        self.model.zero_grad()
        output[:, class_idx].backward(retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients  # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]

        # Compute Grad-CAM++
        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                     (gradients.pow(3) * activations).sum(dim=(2, 3), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num / alpha_denom  # [1, C, H, W]

        weights = (alpha * torch.relu(gradients)).sum(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Upsample to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        return cam.squeeze().detach().cpu().numpy()

class DefectLocalizer:
    """
    Main class for defect localization combining multiple approaches.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Initialize components
        self.attention_encoder = DefectAttentionEncoder().to(device)
        self.patch_detector = PatchAnomalyDetector().to(device)

        # Grad-CAM++ for attention encoder
        self.grad_cam = GradCAMPlusPlus(self.attention_encoder, self.attention_encoder.layer4)

        logger.info("DefectLocalizer initialized")

    def localize_defects(self, image):
        """
        Localize defects in an image.
        image: PIL Image or numpy array [H, W, C]
        Returns: heatmap, bounding_boxes, mask
        """
        # Preprocess image
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            # Convert PIL to tensor
            from torchvision import transforms
            transform = transforms.ToTensor()
            image = transform(image)

        image = image.unsqueeze(0).to(self.device)  # [1, 3, H, W]

        # Get attention features
        with torch.no_grad():
            features, attention_maps = self.attention_encoder(image)

        # Generate Grad-CAM++ heatmap
        gradcam_heatmap = self.grad_cam.generate_heatmap(image)

        # Get patch-level anomaly scores
        anomaly_map, _ = self.patch_detector(image)
        anomaly_map = anomaly_map.squeeze().cpu().numpy()

        # Upsample anomaly map to image size
        anomaly_heatmap = cv2.resize(anomaly_map, (image.shape[3], image.shape[2]),
                                   interpolation=cv2.INTER_LINEAR)

        # Combine heatmaps
        combined_heatmap = self._combine_heatmaps(gradcam_heatmap, anomaly_heatmap, attention_maps)

        # Threshold and extract regions
        thresholded_mask = self._threshold_heatmap(combined_heatmap)
        bounding_boxes = self._extract_bounding_boxes(thresholded_mask)

        return combined_heatmap, bounding_boxes, thresholded_mask

    def _combine_heatmaps(self, gradcam_heatmap, anomaly_heatmap, attention_maps):
        """
        Combine multiple heatmaps using weighted fusion.
        """
        # Normalize heatmaps to [0, 1]
        gradcam_norm = (gradcam_heatmap - gradcam_heatmap.min()) / (gradcam_heatmap.max() - gradcam_heatmap.min() + 1e-8)
        anomaly_norm = (anomaly_heatmap - anomaly_heatmap.min()) / (anomaly_heatmap.max() - anomaly_heatmap.min() + 1e-8)

        # Get attention map (upsample to image size)
        attention_map = attention_maps['fused'].squeeze().cpu().numpy()
        attention_norm = cv2.resize(attention_map, (gradcam_heatmap.shape[1], gradcam_heatmap.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
        attention_norm = (attention_norm - attention_norm.min()) / (attention_norm.max() - attention_norm.min() + 1e-8)

        # Weighted combination
        weights = {'gradcam': 0.4, 'anomaly': 0.4, 'attention': 0.2}
        combined = (weights['gradcam'] * gradcam_norm +
                   weights['anomaly'] * anomaly_norm +
                   weights['attention'] * attention_norm)

        return combined

    def _threshold_heatmap(self, heatmap, threshold_percentile=85):
        """
        Threshold heatmap to create binary mask.
        """
        threshold = np.percentile(heatmap, threshold_percentile)
        mask = (heatmap > threshold).astype(np.uint8)
        return mask

    def _extract_bounding_boxes(self, mask, min_area=100):
        """
        Extract bounding boxes from binary mask.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append([x, y, x+w, y+h])  # [x1, y1, x2, y2]

        return bounding_boxes

    def update_patch_memory(self, healthy_images):
        """
        Update patch detector memory with healthy images.
        healthy_images: list of PIL Images or numpy arrays
        """
        logger.info("Updating patch memory with healthy images...")

        all_patch_features = []
        for img in healthy_images:
            # Preprocess
            if isinstance(img, np.ndarray):
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            else:
                from torchvision import transforms
                transform = transforms.ToTensor()
                img_tensor = transform(img)

            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            # Extract patch features
            with torch.no_grad():
                _, patch_features = self.patch_detector(img_tensor)
                all_patch_features.append(patch_features.flatten(0, 1))  # [num_patches, embed_dim]

        # Concatenate all features
        if all_patch_features:
            healthy_features = torch.cat(all_patch_features, dim=0)
            self.patch_detector.update_memory(healthy_features)
            logger.info(f"Updated memory with {healthy_features.shape[0]} healthy patches")

if __name__ == "__main__":
    # Test the localizer
    localizer = DefectLocalizer()

    # Create dummy image
    test_image = np.random.rand(224, 224, 3).astype(np.float32)

    # Test localization
    heatmap, bboxes, mask = localizer.localize_defects(test_image)

    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Number of bounding boxes: {len(bboxes)}")
    print(f"Mask shape: {mask.shape}")

    print("DefectLocalizer test completed!")
