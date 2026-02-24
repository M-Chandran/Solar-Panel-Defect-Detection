import torch
import torch.nn as nn
import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Any, Optional
from .image_processing_defect_detector import ImageProcessingDefectDetector
from .localization import DefectLocalizer
from .classification import DefectClassifier
from .explainability import ExplainableAIDefectDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridDefectDetector:
    """
    Advanced hybrid detector combining rule-based OpenCV processing with deep learning.
    Implements confidence-weighted fusion of multiple detection methods.
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Initialize detection components
        self.rule_based_detector = ImageProcessingDefectDetector()
        self.deep_localizer = DefectLocalizer(device=device)
        self.deep_classifier = None  # Will be initialized when model is available
        self.explainer = None  # Will be initialized when model is available

        # Fusion weights (can be learned or tuned)
        self.fusion_weights = {
            'rule_based': 0.6,
            'deep_learning': 0.4,
            'confidence_threshold': 0.7
        }

        # Feature fusion parameters
        self.feature_fusion_enabled = True
        self.confidence_weighting = True

        logger.info("HybridDefectDetector initialized with rule-based + deep learning fusion")

    def set_deep_learning_model(self, model, classifier_model=None):
        """
        Set the deep learning model for hybrid detection.

        Args:
            model: Trained feature extraction model (e.g., MoCo + Attention)
            classifier_model: Optional trained classifier
        """
        self.deep_model = model.to(self.device)
        self.deep_model.eval()

        if classifier_model:
            self.deep_classifier = DefectClassifier()
            self.deep_classifier.is_trained = True  # Assume pre-trained
            # Note: In practice, you'd load the trained classifier

        # Initialize explainer
        self.explainer = ExplainableAIDefectDetector(model, self.device)

        logger.info("Deep learning model set for hybrid detection")

    def detect_defects_hybrid(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform hybrid defect detection combining multiple methods.

        Args:
            image: Input image as numpy array [H, W, C]

        Returns:
            Comprehensive detection results with fused predictions
        """
        # 1. Rule-based detection
        rule_results = self.rule_based_detector.detect_defects(image)

        # 2. Deep learning detection (if available)
        deep_results = None
        if hasattr(self, 'deep_model'):
            deep_results = self._deep_learning_detection(image)

        # 3. Feature fusion
        if deep_results and self.feature_fusion_enabled:
            fused_results = self._fuse_detection_results(rule_results, deep_results)
        else:
            fused_results = rule_results

        # 4. Enhanced explanations (if explainer available)
        if self.explainer:
            explanations = self._generate_hybrid_explanations(image, fused_results)
            fused_results['explanations'] = explanations

        # 5. Generate recommendations
        fused_results['recommendations'] = self._generate_advanced_recommendations(fused_results)

        return fused_results

    def _deep_learning_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform deep learning-based defect detection.
        """
        try:
            # Get localization results
            heatmap, bboxes, mask = self.deep_localizer.localize_defects(image)

            # Convert mask to defects format
            defects = []
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)

                defect = {
                    'id': i + 1,
                    'type': 'unknown',  # Will be classified if classifier available
                    'location': self._get_location_from_bbox(bbox, image.shape),
                    'bounding_box': bbox,
                    'area': area,
                    'severity': 'unknown',
                    'confidence': 0.8,  # Default confidence for deep method
                    'description': 'Detected by deep learning localization'
                }

                # Classify defect if classifier available
                if self.deep_classifier:
                    defect_region = image[y1:y2, x1:x2]
                    if defect_region.size > 0:
                        try:
                            defect_types, confidences = self.deep_classifier.predict(
                                self.deep_model, [defect_region]
                            )
                            defect['type'] = defect_types[0] if defect_types else 'unknown'
                            defect['confidence'] = confidences[0] if confidences else 0.5
                        except Exception as e:
                            logger.warning(f"Deep classification failed: {e}")

                defects.append(defect)

            return {
                'defects': defects,
                'defect_count': len(defects),
                'heatmap': heatmap,
                'bounding_boxes': bboxes,
                'mask': mask,
                'method': 'deep_learning'
            }

        except Exception as e:
            logger.error(f"Deep learning detection failed: {e}")
            return None

    def _fuse_detection_results(self, rule_results: Dict, deep_results: Dict) -> Dict:
        """
        Fuse results from rule-based and deep learning methods.
        """
        fused_defects = []
        used_deep_indices = set()

        # Process rule-based defects
        for rule_defect in rule_results['defects']:
            rule_bbox = rule_defect['bounding_box']
            rule_center = self._get_bbox_center(rule_bbox)

            # Find overlapping deep learning detections
            best_match = None
            best_iou = 0.0

            for i, deep_defect in enumerate(deep_results['defects']):
                if i in used_deep_indices:
                    continue

                deep_bbox = deep_defect['bounding_box']
                iou = self._calculate_iou(rule_bbox, deep_bbox)

                if iou > 0.3 and iou > best_iou:  # IoU threshold
                    best_match = (i, deep_defect)
                    best_iou = iou

            if best_match:
                # Fuse the detections
                deep_idx, deep_defect = best_match
                fused_defect = self._fuse_single_defects(rule_defect, deep_defect)
                used_deep_indices.add(deep_idx)
            else:
                # Use rule-based defect with reduced confidence
                fused_defect = rule_defect.copy()
                fused_defect['confidence'] *= 0.8  # Reduce confidence for unmatched detections
                fused_defect['method'] = 'rule_based'

            fused_defects.append(fused_defect)

        # Add remaining deep learning detections
        for i, deep_defect in enumerate(deep_results['defects']):
            if i not in used_deep_indices:
                fused_defect = deep_defect.copy()
                fused_defect['confidence'] *= 0.7  # Reduce confidence for unmatched detections
                fused_defect['method'] = 'deep_learning'
                fused_defects.append(fused_defect)

        # Create fused heatmap
        fused_heatmap = self._fuse_heatmaps(
            rule_results.get('heatmap', np.zeros_like(rule_results['original_image'])),
            deep_results.get('heatmap', np.zeros_like(rule_results['original_image']))
        )

        return {
            'defects': fused_defects,
            'defect_count': len(fused_defects),
            'heatmap': fused_heatmap,
            'bounding_boxes': [d['bounding_box'] for d in fused_defects],
            'original_image': rule_results['original_image'],
            'method': 'hybrid_fusion'
        }

    def _fuse_single_defects(self, rule_defect: Dict, deep_defect: Dict) -> Dict:
        """
        Fuse two defect detections into one.
        """
        # Confidence-weighted fusion
        rule_conf = rule_defect['confidence']
        deep_conf = deep_defect['confidence']

        if self.confidence_weighting:
            total_conf = rule_conf + deep_conf
            rule_weight = rule_conf / total_conf if total_conf > 0 else 0.5
            deep_weight = deep_conf / total_conf if total_conf > 0 else 0.5
        else:
            rule_weight = self.fusion_weights['rule_based']
            deep_weight = self.fusion_weights['deep_learning']

        # Fuse bounding boxes (weighted average)
        rule_bbox = np.array(rule_defect['bounding_box'])
        deep_bbox = np.array(deep_defect['bounding_box'])
        fused_bbox = (rule_weight * rule_bbox + deep_weight * deep_bbox).astype(int).tolist()

        # Fuse other properties
        fused_defect = {
            'id': rule_defect['id'],  # Keep rule-based ID
            'type': deep_defect['type'] if deep_conf > rule_conf else rule_defect['type'],
            'location': rule_defect['location'],  # Keep rule-based location
            'bounding_box': fused_bbox,
            'area': int(rule_weight * rule_defect['area'] + deep_weight * deep_defect['area']),
            'severity': rule_defect['severity'],  # Keep rule-based severity
            'confidence': max(rule_conf, deep_conf),  # Take higher confidence
            'description': f"Fused detection: {rule_defect['type']} + {deep_defect['type']}",
            'method': 'hybrid_fusion'
        }

        return fused_defect

    def _fuse_heatmaps(self, rule_heatmap: np.ndarray, deep_heatmap: np.ndarray) -> np.ndarray:
        """
        Fuse heatmaps from different methods.
        """
        # Normalize heatmaps
        rule_norm = (rule_heatmap - rule_heatmap.min()) / (rule_heatmap.max() - rule_heatmap.min() + 1e-8)
        deep_norm = (deep_heatmap - deep_heatmap.min()) / (deep_heatmap.max() - deep_heatmap.min() + 1e-8)

        # Weighted fusion
        fused = (self.fusion_weights['rule_based'] * rule_norm +
                self.fusion_weights['deep_learning'] * deep_norm)

        return fused

    def _generate_hybrid_explanations(self, image: np.ndarray, results: Dict) -> Dict:
        """
        Generate explanations for hybrid detection results.
        """
        try:
            # Convert image to tensor
            if isinstance(image, np.ndarray):
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                image_tensor = image_tensor.unsqueeze(0).to(self.device)

            # Generate deep learning explanations
            explanations = self.explainer.generate_explanations(image_tensor)

            # Add hybrid-specific explanations
            explanations['fusion_method'] = 'confidence_weighted_fusion'
            explanations['detection_methods'] = ['rule_based_opencv', 'deep_learning_localization']

            # Analyze fusion quality
            fusion_analysis = self._analyze_fusion_quality(results)
            explanations['fusion_analysis'] = fusion_analysis

            return explanations

        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}")
            return {'error': str(e)}

    def _analyze_fusion_quality(self, results: Dict) -> Dict:
        """
        Analyze the quality of detection fusion.
        """
        analysis = {
            'total_defects': results['defect_count'],
            'fusion_methods': [],
            'confidence_distribution': [],
            'method_agreement': 0.0
        }

        method_counts = {'rule_based': 0, 'deep_learning': 0, 'hybrid_fusion': 0}

        for defect in results['defects']:
            method = defect.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
            analysis['confidence_distribution'].append(defect['confidence'])

        analysis['fusion_methods'] = method_counts

        # Calculate method agreement (simplified)
        if method_counts['hybrid_fusion'] > 0:
            agreement = method_counts['hybrid_fusion'] / sum(method_counts.values())
            analysis['method_agreement'] = agreement

        return analysis

    def _generate_advanced_recommendations(self, results: Dict) -> List[str]:
        """
        Generate advanced recommendations based on hybrid detection results.
        """
        recommendations = []

        defects = results.get('defects', [])
        defect_count = len(defects)

        # Analyze defect patterns
        defect_types = [d['type'] for d in defects]
        severities = [d['severity'] for d in defects]
        confidences = [d['confidence'] for d in defects]

        # High-confidence detections
        high_conf_defects = [d for d in defects if d['confidence'] > 0.8]

        if defect_count == 0:
            recommendations.append("✅ No defects detected. Panel appears healthy.")
            recommendations.append("📅 Schedule routine inspection in 6-12 months.")
        else:
            # Defect severity analysis
            high_severity = sum(1 for s in severities if s == 'High')
            if high_severity > 0:
                recommendations.append(f"🚨 {high_severity} high-severity defects detected. Immediate inspection recommended.")

            # Defect type analysis
            if 'crack' in defect_types:
                recommendations.append("🔍 Micro-cracks detected. Consider electroluminescence testing.")
            if 'hotspot' in defect_types:
                recommendations.append("🌡️ Hotspots identified. Monitor temperature and electrical performance.")
            if 'burn_damage' in defect_types:
                recommendations.append("🔥 Burn damage present. Evaluate structural integrity.")

            # Confidence analysis
            avg_confidence = np.mean(confidences) if confidences else 0
            if avg_confidence > 0.8:
                recommendations.append("🎯 High-confidence detections. Results are reliable.")
            elif avg_confidence < 0.6:
                recommendations.append("⚠️ Low-confidence detections. Consider manual verification.")

            # Pattern analysis
            if defect_count > 3:
                recommendations.append("📊 Multiple defects detected. Comprehensive panel assessment recommended.")

        return recommendations

    # Utility methods
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _get_bbox_center(self, bbox: List[int]) -> Tuple[int, int]:
        """Get center coordinates of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _get_location_from_bbox(self, bbox: List[int], image_shape: Tuple[int, int, int]) -> str:
        """Convert bounding box to location description."""
        height, width = image_shape[:2]
        center_x, center_y = self._get_bbox_center(bbox)

        if center_x < width/2 and center_y < height/2:
            return "Top-left region"
        elif center_x >= width/2 and center_y < height/2:
            return "Top-right region"
        elif center_x < width/2 and center_y >= height/2:
            return "Bottom-left region"
        else:
            return "Bottom-right region"

if __name__ == "__main__":
    # Test the hybrid detector
    print("Testing HybridDefectDetector...")

    detector = HybridDefectDetector()

    # Create test image
    test_image = np.random.rand(224, 224, 3).astype(np.float32)

    # Test rule-based only
    results = detector.detect_defects_hybrid(test_image)
    print(f"Hybrid detection found {results['defect_count']} defects")
    print(f"Recommendations: {results['recommendations']}")

    print("HybridDefectDetector test completed!")
