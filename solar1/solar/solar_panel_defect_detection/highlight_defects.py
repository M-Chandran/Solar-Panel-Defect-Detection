"""
Enhanced Defect Highlighting for Solar Panel Images
Comprehensive defect detection with multiple types and clear visualization.
"""

import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def suppress_grid_lines(gray_image):
    """
    Suppress solar panel grid lines to avoid false positive crack detections.
    """
    h, w = gray_image.shape[:2]
    
    # Detect horizontal grid lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 15), 1))
    horizontal_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Detect vertical grid lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 15)))
    vertical_lines = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combine grid lines
    grid_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    # Dilate to cover grid line edges
    grid_mask = cv2.dilate(grid_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    
    return grid_mask


def get_optimal_grid_size(image_size):
    """
    Get optimal grid size based on image dimensions.
    For smaller images, use smaller grid to maintain meaningful cell sizes.
    """
    min_cell_size = 25  # Minimum cell size in pixels
    
    # Calculate max rows/cols based on minimum cell size
    h, w = image_size[:2]
    
    # Use at least 3x3 grid, at most 10x10
    max_rows = max(3, min(10, h // min_cell_size))
    max_cols = max(3, min(10, w // min_cell_size))
    
    return max_rows, max_cols


def detect_defects_enhanced(image, sensitivity=0.7):
    """
    Enhanced defect detection with multiple defect types.
    """
    h, w = image.shape[:2]
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Get grid mask
    grid_mask = suppress_grid_lines(gray)
    
    # Calculate overall image statistics
    global_mean = float(np.mean(gray))
    global_std = float(np.std(gray))
    
    # Get optimal grid size based on image size
    rows, cols = get_optimal_grid_size(gray.shape)
    cell_h, cell_w = h // rows, w // cols
    
    # For small images, use more lenient thresholds
    is_small_image = h < 300 or w < 300
    
    threshold_factor = 1.0 - (sensitivity * 0.3)
    if is_small_image:
        threshold_factor *= 0.8  # More sensitive for small images
    
    defects = []
    
    for row in range(rows):
        for col in range(cols):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            
            cell = gray[y1:y2, x1:x2]
            
            # Calculate statistics
            mean_int = float(np.mean(cell))
            std_int = float(np.std(cell))
            min_int = float(np.min(cell))
            max_int = float(np.max(cell))
            
            # Grid ratio
            grid_cell = grid_mask[y1:y2, x1:x2]
            grid_ratio = float(np.sum(grid_cell > 0) / grid_cell.size) if grid_cell.size > 0 else 0
            
            # ====== DEFECT DETECTION ======
            
            # 1. BURN MARK: Very dark regions
            burn_mean_thresh = 90 * threshold_factor
            burn_std_thresh = 25 * threshold_factor if is_small_image else 30 * threshold_factor
            burn_min_thresh = 40 * threshold_factor if is_small_image else 50 * threshold_factor
            
            mean_deviation = global_mean - mean_int
            
            if (mean_int < burn_mean_thresh and 
                std_int > burn_std_thresh and 
                min_int < burn_min_thresh and
                mean_deviation > 15):
                
                conf1 = min(1.0, max(0, (burn_mean_thresh - mean_int) / burn_mean_thresh))
                conf2 = min(1.0, max(0, (std_int - burn_std_thresh) / 80))
                conf3 = min(1.0, max(0, (burn_min_thresh - min_int) / burn_min_thresh * 0.5))
                confidence = min(1.0, conf1 + conf2 + conf3)
                
                if confidence > 0.4:
                    defects.append({
                        'bbox': (x1, y1, cell_w, cell_h),
                        'type': 'BURN_MARK',
                        'confidence': confidence,
                        'severity': 'CRITICAL' if confidence > 0.7 else 'HIGH' if confidence > 0.55 else 'MODERATE',
                        'row': row,
                        'col': col,
                        'metrics': {'mean': mean_int, 'std': std_int, 'min': min_int}
                    })
                    continue
            
            # 2. HOTSPOT: Very bright regions
            hotspot_mean_thresh = 175 * threshold_factor if is_small_image else 190 * threshold_factor
            hotspot_std_thresh = 25 * threshold_factor if is_small_image else 30 * threshold_factor
            hotspot_max_thresh = 220 * threshold_factor if is_small_image else 230 * threshold_factor
            
            mean_excess = mean_int - global_mean
            
            if (mean_int > hotspot_mean_thresh and 
                std_int > hotspot_std_thresh and 
                max_int > hotspot_max_thresh and
                mean_excess > 20):
                
                excess_val = min(1.0, max(0, (mean_int - 175) / 65))
                std_val = min(1.0, max(0, (std_int - 25) / 100))
                max_val = min(1.0, max(0, (max_int - 220) / 25 * 0.3))
                confidence = min(1.0, excess_val + std_val + max_val)
                
                if confidence > 0.4:
                    defects.append({
                        'bbox': (x1, y1, cell_w, cell_h),
                        'type': 'HOTSPOT',
                        'confidence': confidence,
                        'severity': 'CRITICAL' if confidence > 0.7 else 'HIGH' if confidence > 0.55 else 'MODERATE',
                        'row': row,
                        'col': col,
                        'metrics': {'mean': mean_int, 'std': std_int, 'max': max_int}
                    })
                    continue
            
            # 3. CRACK: High edge density - exclude grid areas
            if grid_ratio < 0.15:
                edges = cv2.Canny(cell, 40, 120)
                edge_ratio = float(np.sum(edges > 0) / edges.size) if edges.size > 0 else 0
                
                crack_edge_thresh = 0.12 * threshold_factor if is_small_image else 0.15 * threshold_factor
                if edge_ratio > crack_edge_thresh:
                    confidence = min(1.0, (edge_ratio - crack_edge_thresh) / crack_edge_thresh + 0.4)
                    
                    if confidence > 0.45:
                        defects.append({
                            'bbox': (x1, y1, cell_w, cell_h),
                            'type': 'CRACK',
                            'confidence': confidence,
                            'severity': 'HIGH' if confidence > 0.65 else 'MODERATE',
                            'row': row,
                            'col': col,
                            'metrics': {'edge_ratio': edge_ratio}
                        })
                        continue
            
            # 4. DISCOLORATION: Color variation (limited)
            if len(image.shape) == 3 and len([d for d in defects if d['type'] == 'DISCOLORATION']) < 15:
                cell_color = image[y1:y2, x1:x2]
                lab = cv2.cvtColor(cell_color, cv2.COLOR_RGB2LAB)
                color_std = float(np.std(lab[:,:,1]) + np.std(lab[:,:,2]))
                
                discolor_thresh = 25 * threshold_factor if is_small_image else 35 * threshold_factor
                if color_std > discolor_thresh and mean_int < 160:
                    confidence = min(1.0, (color_std - discolor_thresh) / discolor_thresh * 0.5 + 0.4)
                    
                    if confidence > 0.4:
                        defects.append({
                            'bbox': (x1, y1, cell_w, cell_h),
                            'type': 'DISCOLORATION',
                            'confidence': confidence,
                            'severity': 'MODERATE',
                            'row': row,
                            'col': col,
                            'metrics': {'color_std': color_std}
                        })
    
    return defects


def highlight_defects(image_path, output_path='highlighted_defects.png', sensitivity=0.7):
    """
    Main function to detect and highlight defects in solar panel image.
    """
    print(f"\n{'='*60}")
    print("SOLAR PANEL DEFECT DETECTION")
    print(f"{'='*60}")
    print(f"Processing: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Image size: {image.shape[1]} x {image.shape[0]} pixels")
    
    # Detect defects
    print(f"Detecting defects with sensitivity={sensitivity}...")
    defects = detect_defects_enhanced(image, sensitivity=sensitivity)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.imshow(image)
    
    # Color scheme for defect types
    colors = {
        'BURN_MARK': '#FF0000',      # Red
        'HOTSPOT': '#FF6600',        # Orange
        'CRACK': '#FF00FF',          # Magenta
        'DISCOLORATION': '#FFFF00'   # Yellow
    }
    
    # Severity background colors
    severity_bg = {
        'CRITICAL': (1.0, 0.0, 0.0, 0.45),
        'HIGH': (1.0, 0.4, 0.0, 0.4),
        'MODERATE': (1.0, 1.0, 0.0, 0.35)
    }
    
    found_types = set()
    
    if not defects:
        print("\n✓ No defects detected.")
    else:
        print(f"\n✓ Found {len(defects)} defect(s)")
        
        # Sort by severity
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MODERATE': 2}
        defects.sort(key=lambda x: (severity_order.get(x['severity'], 3), -x['confidence']))
        
        print(f"\n{'─'*60}")
        print("DETECTED DEFECTS:")
        print(f"{'─'*60}")
        
        for i, defect in enumerate(defects, 1):
            x, y, cell_w, cell_h = defect['bbox']
            dtype = defect['type']
            conf = defect['confidence']
            sev = defect['severity']
            
            found_types.add(dtype)
            
            color = colors.get(dtype, '#FF0000')
            bg_color = severity_bg.get(sev, (1, 1, 0, 0.2))
            
            # Draw background
            bg_rect = Rectangle((x, y), cell_w, cell_h, 
                              facecolor=bg_color, edgecolor='none', zorder=1)
            ax.add_patch(bg_rect)
            
            # Draw border
            border_width = 5 if sev == 'CRITICAL' else 4
            rect = Rectangle((x, y), cell_w, cell_h, 
                           linewidth=border_width, 
                           edgecolor=color, 
                           facecolor='none', zorder=2)
            ax.add_patch(rect)
            
            # Add label
            label = f"{dtype}\n{conf:.0%}"
            ax.text(x + 5, y + 20, label, 
                   fontsize=9, weight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor=color, alpha=0.95,
                            edgecolor='white', linewidth=2), zorder=3)
            
            print(f"  {i}. {dtype} at cell [{defect['row']},{defect['col']}]")
            print(f"     Severity: {sev}, Confidence: {conf:.1%}")
    
    # Determine overall condition
    critical_count = sum(1 for d in defects if d['severity'] == 'CRITICAL')
    high_count = sum(1 for d in defects if d['severity'] == 'HIGH')
    
    if critical_count > 0:
        condition = "CRITICAL - FAILED"
        title_color = 'red'
    elif high_count > 0:
        condition = "DEGRADED"
        title_color = 'orange'
    elif len(defects) > 0:
        condition = "MINOR DEFECTS"
        title_color = '#FFCC00'
    else:
        condition = "NORMAL"
        title_color = 'green'
    
    # Set title
    title = f"Solar Panel Defect Detection | Condition: {condition}"
    if defects:
        title += f" | Defects: {len(defects)}"
    
    ax.set_title(title, fontsize=14, weight='bold', color=title_color, pad=20)
    ax.set_axis_off()
    
    # Add legend
    if found_types:
        legend_elements = []
        for dtype in ['BURN_MARK', 'HOTSPOT', 'CRACK', 'DISCOLORATION']:
            if dtype in found_types:
                legend_elements.append(mpatches.Patch(facecolor=colors[dtype], 
                                                     edgecolor='white',
                                                     label=dtype.replace('_', ' '), alpha=0.8))
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', 
                    fontsize=11, framealpha=0.95)
    
    # Draw grid
    h, w = image.shape[:2]
    rows, cols = get_optimal_grid_size(image.shape)
    cell_h, cell_w = h // rows, w // cols
    
    for row in range(rows + 1):
        ax.axhline(row * cell_h, color='white', alpha=0.2, linewidth=0.5)
    for col in range(cols + 1):
        ax.axvline(col * cell_w, color='white', alpha=0.2, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n{'─'*60}")
    print(f"Saved: {output_path}")
    print(f"{'─'*60}")
    
    return {
        'defects': defects,
        'condition': condition,
        'count': len(defects),
        'defect_types': list(found_types)
    }


def batch_process(input_dir, output_dir='highlighted'):
    """
    Process all images in a directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nFound {len(image_files)} images to process")
    
    results = {}
    for img_file in image_files:
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, f"highlighted_{img_file}")
        
        try:
            result = highlight_defects(input_path, output_path)
            results[img_file] = result
            print(f"✓ Processed: {img_file}")
        except Exception as e:
            print(f"✗ Failed: {img_file} - {e}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Highlight solar panel defects')
    parser.add_argument('image', nargs='?', default='solar.jpg', 
                       help='Image file or directory')
    parser.add_argument('--output', '-o', default='highlighted_defects.png',
                       help='Output file path')
    parser.add_argument('--sensitivity', '-s', type=float, default=0.7,
                       help='Detection sensitivity (0-1)')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process directory in batch mode')
    
    args = parser.parse_args()
    
    if args.batch or os.path.isdir(args.image):
        results = batch_process(args.image, args.output)
        print(f"\nBatch processing complete. Processed {len(results)} images.")
    else:
        result = highlight_defects(args.image, args.output, args.sensitivity)
        print(f"\nCondition: {result['condition']}")
        print(f"Defects found: {result['count']}")
