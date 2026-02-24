import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessingDefectDetector:
    """
    Rule-based defect detection using image processing pipeline.
    No deep learning models or pre-trained datasets used.
    """

    def __init__(self, sensitivity: float = 0.6):
        self.min_defect_area = 40
        self.max_defect_area = 30000
        self.irregularity_threshold = 0.25
        self.intensity_deviation_threshold = 12
        self.contrast_threshold = 35
        self.dark_region_threshold = 110
        self.set_sensitivity(sensitivity)

    def set_sensitivity(self, sensitivity: float) -> None:
        """
        Update detector sensitivity in [0.0, 1.0].
        Higher values increase recall (more detections, more false positives).
        """
        self.sensitivity = float(np.clip(sensitivity, 0.0, 1.0))

    def detect_defects(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Main defect detection pipeline.

        Args:
            image: Input image as numpy array (H, W, C) or (H, W)

        Returns:
            Dictionary containing defect information, heatmap, and bounding boxes
        """
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")

        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Store original image
        original_image = image.copy()

        # Step 1: Image Preprocessing
        processed_image, gray, enhanced = self._preprocess_image(image)

        # Step 2: Grid Line Suppression
        grid_suppressed = self._suppress_grid_lines(processed_image)

        # Step 3: Defect Region Detection
        contours, defect_mask = self._detect_defect_regions(grid_suppressed, gray, enhanced)

        # Step 4: Defect Classification and Analysis
        defects = self._classify_defects(contours, original_image, defect_mask)

        # Confidence gating to reduce grid-pattern false positives.
        confidence_threshold = float(np.clip(0.82 - 0.22 * self.sensitivity, 0.55, 0.82))
        defects = [d for d in defects if d['confidence'] >= confidence_threshold]

        # Rebuild mask from accepted defects to keep outputs consistent.
        defect_mask = np.zeros_like(defect_mask)
        for defect in defects:
            x1, y1, x2, y2 = defect['bounding_box']
            cv2.rectangle(defect_mask, (x1, y1), (x2, y2), 255, -1)

        # Step 5: Create visualization
        heatmap = self._create_heatmap(original_image, defect_mask, defects)

        bounding_boxes = [d['bounding_box'] for d in defects]

        return {
            'defects': defects,
            'defect_count': len(defects),
            'heatmap': heatmap,
            'bounding_boxes': bounding_boxes,
            'defect_mask': defect_mask,
            'original_image': original_image,
            'processed_image': processed_image
        }

    def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Step 1: Image Preprocessing
        - Convert to Grayscale
        - Apply Gaussian Blur
        - Use Adaptive Thresholding
        - Apply Morphological Opening to remove grid lines
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Enhance local contrast for low-contrast defects
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced = cv2.bilateralFilter(enhanced, d=7, sigmaColor=35, sigmaSpace=35)

        # Adaptive threshold for local anomalies
        adaptive_mask = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
        )

        # Black-hat to reveal dark defects on bright background
        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, blackhat_kernel)
        _, blackhat_mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        combined = cv2.bitwise_or(adaptive_mask, blackhat_mask)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        return combined, gray, enhanced

    def _suppress_grid_lines(self, image: np.ndarray) -> np.ndarray:
        """
        Step 2: Grid Line Suppression
        - Detect straight lines using Hough Transform
        - Remove them using masking
        """
        h, w = image.shape[:2]
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 20), 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h // 20)))

        horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        vertical_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        grid_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        grid_mask = cv2.dilate(grid_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        # Keep non-grid candidates
        suppressed = cv2.bitwise_and(image, cv2.bitwise_not(grid_mask))
        suppressed = cv2.morphologyEx(
            suppressed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
        )
        # Fallback: if suppression removes too much signal, keep original mask.
        original_nonzero = int(np.count_nonzero(image))
        suppressed_nonzero = int(np.count_nonzero(suppressed))
        if original_nonzero > 0 and suppressed_nonzero < int(0.20 * original_nonzero):
            return image
        return suppressed

    def _detect_defect_regions(self, image: np.ndarray, gray: np.ndarray,
                               enhanced: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Step 3: Defect Region Detection
        - Use Contour Detection
        - Filter contours based on criteria
        """
        # Find contours on candidate mask
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create defect mask
        defect_mask = np.zeros_like(image)

        filtered_contours = []
        total_pixels = gray.shape[0] * gray.shape[1]
        min_area = max(20, int(self.min_defect_area * (1.0 - 0.6 * self.sensitivity)))
        max_area = min(int(0.20 * total_pixels), int(self.max_defect_area * (1.0 + 0.5 * self.sensitivity)))

        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Filter based on minimum area
            if area < min_area or area > max_area:
                continue

            # Calculate shape irregularity (circularity)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                irregularity = 1 - circularity  # Higher values = more irregular
            else:
                irregularity = 1.0

            # Filter based on irregularity (must be irregular, not circular)
            if irregularity < self.irregularity_threshold:
                continue

            # Evaluate contour on real intensity image (not binary mask)
            x, y, w, h = cv2.boundingRect(contour)
            roi_gray = gray[y:y+h, x:x+w]
            roi_enh = enhanced[y:y+h, x:x+w]
            if roi_gray.size == 0:
                continue

            local_mask = np.zeros((h, w), dtype=np.uint8)
            shifted = contour - [x, y]
            cv2.drawContours(local_mask, [shifted], -1, 255, -1)
            defect_pixels = roi_gray[local_mask > 0]
            if defect_pixels.size == 0:
                continue

            mean_intensity = float(np.mean(defect_pixels))
            std_intensity = float(np.std(defect_pixels))
            p10 = float(np.percentile(defect_pixels, 10))
            p90 = float(np.percentile(defect_pixels, 90))
            local_contrast = p90 - p10

            # Compare defect region against nearby ring context
            ring_mask = cv2.dilate(local_mask, np.ones((9, 9), dtype=np.uint8), iterations=1)
            ring_mask = cv2.subtract(ring_mask, local_mask)
            neighborhood = roi_enh[ring_mask > 0]
            if neighborhood.size > 0:
                neighborhood_mean = float(np.mean(neighborhood))
                intensity_gap = abs(neighborhood_mean - mean_intensity)
            else:
                intensity_gap = 0.0

            # Minimum anomaly score with sensitivity control
            anomaly_score = 0.45 * (std_intensity / 45.0) + 0.35 * (local_contrast / 95.0) + 0.20 * (intensity_gap / 60.0)
            anomaly_threshold = max(0.18, 0.40 - 0.22 * self.sensitivity)
            if anomaly_score < anomaly_threshold:
                continue

            if std_intensity < max(5.0, self.intensity_deviation_threshold * (1.0 - 0.45 * self.sensitivity)):
                continue

            filtered_contours.append(contour)

        filtered_contours = self._merge_overlapping_contours(filtered_contours, gray.shape)
        for contour in filtered_contours:
            cv2.drawContours(defect_mask, [contour], -1, 255, -1)

        return filtered_contours, defect_mask

    def _merge_overlapping_contours(self, contours: List[np.ndarray],
                                    image_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Merge overlapping/nearby contours to avoid fragmented detections.
        """
        if not contours:
            return []

        mask = np.zeros(image_shape, dtype=np.uint8)
        for contour in contours:
            cv2.drawContours(mask, [contour], -1, 255, -1)

        # Mild dilation merges close fragments belonging to the same defect.
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, merge_kernel, iterations=1)
        merged_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return merged_contours

    def _classify_defects(self, contours: List[np.ndarray], original_image: np.ndarray,
                         defect_mask: np.ndarray) -> List[Dict[str, Any]]:
        """
        Step 4: Defect Classification (Rule-Based)
        Classify defects using pixel properties
        """
        defects = []

        for i, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            roi = original_image[y:y+h, x:x+w]

            if roi.size == 0:
                continue

            # Extract features for classification
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

            # Calculate statistics from gray ROI
            mean_intensity = np.mean(gray_roi)
            std_intensity = np.std(gray_roi)
            min_intensity = np.min(gray_roi)
            max_intensity = np.max(gray_roi)

            # Calculate contrast (difference between dark and light areas)
            contrast = max_intensity - min_intensity

            # Calculate shape properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            aspect_ratio = w / h if h > 0 else 1.0
            x2, y2 = x + w, y + h
            bbox_area = max(1, w * h)
            extent = float(area / bbox_area)
            hull = cv2.convexHull(contour)
            hull_area = max(1.0, cv2.contourArea(hull))
            solidity = float(area / hull_area)

            # Rule-based classification
            defect_type, confidence = self._classify_defect_type(
                mean_intensity, std_intensity, contrast, aspect_ratio, area, perimeter, extent, solidity
            )

            # Calculate severity
            severity_score = self._calculate_severity(area, contrast, std_intensity, original_image.shape)

            # Determine location description
            location = self._get_location_description(x, y, w, h, original_image.shape)

            defect = {
                'id': i + 1,
                'type': defect_type,
                'location': location,
                'bounding_box': [x, y, x2, y2],
                'area': area,
                'severity': severity_score,
                'confidence': confidence,
                'description': self._generate_description(defect_type, severity_score, confidence)
            }

            defects.append(defect)

        return defects

    def _classify_defect_type(self, mean_intensity: float, std_intensity: float,
                            contrast: float, aspect_ratio: float, area: float,
                            perimeter: float, extent: float, solidity: float) -> Tuple[str, float]:
        """
        Rule-based defect type classification
        """
        elongation = max(aspect_ratio, 1.0 / max(aspect_ratio, 1e-8))
        crack_score = (elongation / 5.5) + ((1.0 - extent) * 1.4) + ((1.0 - solidity) * 0.9)

        # Crack: elongated and sparse geometry
        if crack_score > 1.1 and perimeter > 45:
            confidence = float(np.clip(0.55 + 0.22 * crack_score, 0.45, 0.95))
            return 'Crack', confidence

        # Hotspot: compact, dark, high contrast
        if contrast > 90 and mean_intensity < 95 and area < 1200 and extent > 0.35:
            confidence = float(np.clip(0.50 + 0.30 * (contrast / 140.0) + 0.20 * (1.0 - mean_intensity / 120.0), 0.45, 0.97))
            return 'Hotspot', confidence

        # Burn Damage: broader dark/charred irregular areas
        if std_intensity > 18 and mean_intensity < 130 and area > 120 and solidity < 0.92:
            confidence = float(np.clip(0.48 + 0.25 * (std_intensity / 45.0) + 0.25 * (1.0 - mean_intensity / 160.0), 0.45, 0.93))
            return 'Burn Damage', confidence

        # Corrosion: rough texture with moderate darkness and irregular shape
        if 35 < contrast < 145 and std_intensity > 16 and solidity < 0.95:
            confidence = float(np.clip(0.45 + 0.25 * (contrast / 120.0) + 0.20 * (std_intensity / 45.0), 0.45, 0.88))
            return 'Corrosion', confidence

        # Fallback
        confidence = float(np.clip(0.42 + 0.25 * (contrast / 120.0), 0.42, 0.78))
        return 'Burn Damage', confidence

    def _calculate_severity(self, area: float, contrast: float, std_intensity: float,
                          image_shape: Tuple[int, int, int]) -> str:
        """
        Calculate defect severity based on area percentage and characteristics
        """
        # Calculate area percentage
        total_area = image_shape[0] * image_shape[1]
        area_percentage = (area / total_area) * 100

        # Severity factors
        area_factor = min(1.0, area_percentage / 5.0)  # 5% area = max severity
        contrast_factor = min(1.0, contrast / 150.0)
        deviation_factor = min(1.0, std_intensity / 50.0)

        severity_score = (area_factor + contrast_factor + deviation_factor) / 3.0

        if severity_score > 0.7:
            return 'High'
        elif severity_score > 0.4:
            return 'Medium'
        else:
            return 'Low'

    def _get_location_description(self, x: int, y: int, w: int, h: int,
                                image_shape: Tuple[int, int, int]) -> str:
        """
        Generate location description based on position
        """
        height, width = image_shape[:2]
        center_x, center_y = x + w/2, y + h/2

        # Determine quadrant
        if center_x < width/2 and center_y < height/2:
            quadrant = "Top-left"
        elif center_x >= width/2 and center_y < height/2:
            quadrant = "Top-right"
        elif center_x < width/2 and center_y >= height/2:
            quadrant = "Bottom-left"
        else:
            quadrant = "Bottom-right"

        return f"{quadrant} region"

    def _generate_description(self, defect_type: str, severity: str, confidence: float) -> str:
        """
        Generate human-readable description
        """
        confidence_pct = int(confidence * 100)
        descriptions = {
            'Hotspot': f"Potential hotspot detected with {severity.lower()} severity ({confidence_pct}% confidence)",
            'Burn Damage': f"Burn damage identified with {severity.lower()} severity ({confidence_pct}% confidence)",
            'Crack': f"Micro-crack detected with {severity.lower()} severity ({confidence_pct}% confidence)",
            'Corrosion': f"Corrosion-like damage detected with {severity.lower()} severity ({confidence_pct}% confidence)"
        }
        return descriptions.get(defect_type, f"Defect detected with {severity.lower()} severity ({confidence_pct}% confidence)")

    def _create_heatmap(self, original_image: np.ndarray, defect_mask: np.ndarray,
                       defects: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create visualization heatmap overlay
        """
        # Resize defect mask to match original image size
        heatmap = cv2.resize(defect_mask.astype(np.uint8), (original_image.shape[1], original_image.shape[0]))

        # Convert to 3-channel
        heatmap_colored = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)

        # Apply different colors based on defect type
        color_map = {
            'Hotspot': [255, 0, 0],      # Red
            'Burn Damage': [0, 0, 255],  # Blue
            'Crack': [255, 255, 0],      # Yellow
            'Corrosion': [0, 255, 255]   # Cyan
        }

        # Create overlay
        overlay = original_image.copy().astype(np.float32)

        for defect in defects:
            x1, y1, x2, y2 = defect['bounding_box']
            color = color_map.get(defect['type'], [255, 255, 255])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Blend with original
        alpha = 0.7
        heatmap_overlay = cv2.addWeighted(original_image.astype(np.float32), 1-alpha,
                                        overlay, alpha, 0).astype(np.uint8)

        return heatmap_overlay


def analyze_image_with_pipeline(image: np.ndarray) -> Dict[str, Any]:
    """
    Convenience function to analyze image using the processing pipeline
    """
    detector = ImageProcessingDefectDetector()
    return detector.detect_defects(image)


if __name__ == "__main__":
    # Test the detector
    import matplotlib.pyplot as plt

    # Create a synthetic defective image for testing
    test_image = np.ones((224, 224, 3), dtype=np.uint8) * 200  # Light gray background

    # Add some grid lines
    for i in range(0, 224, 32):
        cv2.line(test_image, (0, i), (224, i), (150, 150, 150), 1)  # Horizontal
        cv2.line(test_image, (i, 0), (i, 224), (150, 150, 150), 1)  # Vertical

    # Add a simulated defect (dark irregular region)
    cv2.circle(test_image, (100, 100), 20, (50, 50, 50), -1)
    cv2.ellipse(test_image, (150, 150), (15, 25), 45, 0, 360, (30, 30, 30), -1)

    # Run detection
    detector = ImageProcessingDefectDetector()
    results = detector.detect_defects(test_image)

    print(f"Detected {len(results['defects'])} defects")
    for defect in results['defects']:
        print(f"- {defect['type']} at {defect['location']}: {defect['severity']} severity ({defect['confidence']:.2f})")

    print("Image processing pipeline test completed!")
