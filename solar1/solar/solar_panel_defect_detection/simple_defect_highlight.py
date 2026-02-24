"""
Defect Detection for Solar Panel Images
Works WITHOUT any dataset - uses pure image processing to detect defects in ANY image.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


def detect_defects(image, sensitivity=0.5):
    """
    Region-based defect detection that works on ANY image without needing a dataset.
    
    Args:
        image: RGB image
        sensitivity: Detection sensitivity (0.0-1.0), higher = more strict detection
        
    Returns:
        List of detected defects
    """
    h, w = image.shape[:2]
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Global statistics
    global_mean = np.mean(gray)
    global_std = np.std(gray)
    
    # Use coarser grid to avoid too many detections
    rows = 6
    cols = 10
    cell_h, cell_w = h // rows, w // cols
    
    # Base thresholds (will be adjusted by sensitivity)
    base_threshold = 0.5
    
    # Adjust threshold based on sensitivity
    # Higher sensitivity = higher threshold = stricter detection
    threshold = base_threshold + (sensitivity * 0.3)  # Range: 0.5 to 0.8
    
    defects = []
    
    for row in range(rows):
        for col in range(cols):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            
            cell = gray[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            
            # Cell statistics
            mean_val = np.mean(cell)
            std_val = np.std(cell)
            min_val = np.min(cell)
            max_val = np.max(cell)
            
            # Deviation from global mean
            deviation = global_mean - mean_val
            abs_dev = abs(deviation)
            
            # ====== DEFECT DETECTION ======
            
            # 1. BURN MARK - Significantly darker than surroundings
            if (mean_val < global_mean * 0.75 and 
                std_val > global_std * 0.5 and 
                abs_dev > global_std * 0.6):
                
                confidence = min(1.0, (abs_dev / global_std) * 0.7 + 0.3)
                if confidence > threshold:
                    defects.append({
                        'bbox': (x1, y1, x2-x1, y2-y1),
                        'type': 'BURN_MARK',
                        'confidence': confidence,
                        'severity': 'CRITICAL' if confidence > 0.75 else 'HIGH' if confidence > 0.6 else 'MODERATE',
                        'row': row, 'col': col
                    })
                    continue
            
            # 2. HOTSPOT - Significantly brighter than surroundings
            if (mean_val > global_mean * 1.25 and 
                std_val > global_std * 0.5 and 
                abs_dev > global_std * 0.6):
                
                confidence = min(1.0, (abs_dev / global_std) * 0.7 + 0.3)
                if confidence > threshold:
                    defects.append({
                        'bbox': (x1, y1, x2-x1, y2-y1),
                        'type': 'HOTSPOT',
                        'confidence': confidence,
                        'severity': 'CRITICAL' if confidence > 0.75 else 'HIGH' if confidence > 0.6 else 'MODERATE',
                        'row': row, 'col': col
                    })
                    continue
            
            # 3. CRACK - High edge density
            edges = cv2.Canny(cell, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size
            
            if edge_ratio > 0.15:
                confidence = min(1.0, edge_ratio * 4)
                if confidence > threshold + 0.1:  # Extra strict for cracks
                    defects.append({
                        'bbox': (x1, y1, x2-x1, y2-y1),
                        'type': 'CRACK',
                        'confidence': confidence,
                        'severity': 'HIGH' if confidence > 0.7 else 'MODERATE',
                        'row': row, 'col': col
                    })
    
    return defects


def visualize_defects(image, defects, output_path):
    """Create visualization of detected defects."""
    h, w = image.shape[:2]
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.imshow(image)
    
    # Colors
    colors = {
        'BURN_MARK': '#FF0000',
        'HOTSPOT': '#FF6600', 
        'CRACK': '#FF00FF'
    }
    
    severity_colors = {
        'CRITICAL': (1.0, 0.0, 0.0, 0.5),
        'HIGH': (1.0, 0.4, 0.0, 0.45),
        'MODERATE': (1.0, 1.0, 0.0, 0.4)
    }
    
    found_types = set()
    
    if defects:
        # Sort by severity
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MODERATE': 2}
        defects.sort(key=lambda x: (severity_order.get(x['severity'], 3), -x['confidence']))
        
        for d in defects:
            x, y, bw, bh = d['bbox']
            dtype = d['type']
            conf = d['confidence']
            sev = d['severity']
            
            found_types.add(dtype)
            color = colors.get(dtype, '#FF0000')
            sev_color = severity_colors.get(sev, (1, 1, 0, 0.3))
            
            # Background
            bg = Rectangle((x, y), bw, bh, facecolor=sev_color, edgecolor='none', zorder=1)
            ax.add_patch(bg)
            
            # Border
            border_w = 5 if sev == 'CRITICAL' else 4
            rect = Rectangle((x, y), bw, bh, linewidth=border_w, 
                           edgecolor=color, facecolor='none', zorder=2)
            ax.add_patch(rect)
            
            # Label
            ax.text(x + 5, y + 20, f"{dtype}\n{conf:.0%}", 
                   fontsize=9, weight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=color, 
                            alpha=0.95, edgecolor='white', linewidth=2), zorder=3)
        
        # Condition
        critical = sum(1 for d in defects if d['severity'] == 'CRITICAL')
        high = sum(1 for d in defects if d['severity'] == 'HIGH')
        
        if critical > 0:
            condition = "CRITICAL - FAILED"
            title_color = '#FF0000'
        elif high > 0:
            condition = "DEGRADED"
            title_color = '#FF6600'
        else:
            condition = "MINOR DEFECTS"
            title_color = '#FFCC00'
        
        icon = "⚠️"
    else:
        condition = "NORMAL"
        title_color = '#00AA00'
        icon = "✓"
    
    # Title
    title = f"{icon} Solar Panel: {condition}"
    if defects:
        title += f" | {len(defects)} Defect(s)"
    
    ax.set_title(title, fontsize=16, weight='bold', color=title_color, pad=20)
    ax.set_axis_off()
    
    # Legend
    if found_types:
        legend = [mpatches.Patch(facecolor=colors[t], edgecolor='white', 
                               label=t.replace('_', ' '), alpha=0.8) 
                 for t in ['BURN_MARK', 'HOTSPOT', 'CRACK'] if t in found_types]
        if legend:
            ax.legend(handles=legend, loc='upper right', fontsize=11, framealpha=0.95)
    
    # Subtle grid
    for r in range(7):
        ax.axhline(r * h / 6, color='white', alpha=0.15, linewidth=0.5)
    for c in range(11):
        ax.axvline(c * w / 10, color='white', alpha=0.15, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def main(image_path, output_path='result.png', sensitivity=0.5):
    """Main function - detect defects and create visualization."""
    print(f"\nAnalyzing: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot load image: {image_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}")
    
    # Detect
    print(f"Detecting defects (sensitivity={sensitivity})...")
    defects = detect_defects(img, sensitivity=sensitivity)
    
    print(f"Found: {len(defects)} defect(s)")
    
    # Visualize
    visualize_defects(img, defects, output_path)
    print(f"Saved: {output_path}")
    
    # Condition
    if defects:
        critical = sum(1 for d in defects if d['severity'] == 'CRITICAL')
        high = sum(1 for d in defects if d['severity'] == 'HIGH')
        
        if critical > 0:
            condition = "CRITICAL - FAILED"
        elif high > 0:
            condition = "DEGRADED"
        else:
            condition = "MINOR DEFECTS"
    else:
        condition = "NORMAL"
    
    print(f"Condition: {condition}")
    
    return {
        'defects': defects,
        'condition': condition,
        'count': len(defects)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect defects in ANY image - no dataset needed')
    parser.add_argument('image', help='Image path')
    parser.add_argument('-o', '--output', default='result.png', help='Output path')
    parser.add_argument('-s', '--sensitivity', type=float, default=0.5, 
                       help='Sensitivity 0-1, higher = stricter detection')
    
    args = parser.parse_args()
    main(args.image, args.output, args.sensitivity)
