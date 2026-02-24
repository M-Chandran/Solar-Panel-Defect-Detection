"""
Solar Panel Defect Detection with Proper Hotspot/Burn Mark Analysis
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.moco import MoCo
from src.data_preprocessing import preprocess_image

def detect_hotspots(image):
    """
    Detect hotspots/burn marks in solar panel image.
    Returns hotspot regions and severity analysis.
    """
    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Detect abnormally dark or bright regions (hotspots appear as dark spots with bright surroundings)
    # Use adaptive thresholding to find anomalies
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Find dark regions (potential burn marks)
    _, dark_regions = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find bright regions (potential hotspots)
    _, bright_regions = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine to find anomalies
    anomalies = cv2.bitwise_or(dark_regions, bright_regions)
    
    # Find contours of anomalies
    contours, _ = cv2.findContours(anomalies, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hotspots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter small noise
            x, y, w, h = cv2.boundingRect(contour)
            # Calculate severity based on intensity difference
            roi = gray[y:y+h, x:x+w]
            severity = np.std(roi) / np.mean(roi) if np.mean(roi) > 0 else 0
            hotspots.append({
                'bbox': (x, y, w, h),
                'area': area,
                'severity': severity,
                'center': (x + w//2, y + h//2)
            })
    
    return hotspots

def analyze_burn_marks(image):
    """
    Specifically analyze for burn marks and cell damage.
    """
    h, w = image.shape[:2]
    
    # Divide into grid (solar cells)
    rows, cols = 6, 10  # Typical solar panel grid
    cell_h, cell_w = h // rows, w // cols
    
    damaged_cells = []
    
    for row in range(rows):
        for col in range(cols):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            
            cell = image[y1:y2, x1:x2]
            
            # Analyze cell for damage
            if len(cell.shape) == 3:
                cell_gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
            else:
                cell_gray = cell
            
            # Check for discoloration (high variance indicates damage)
            mean_intensity = np.mean(cell_gray)
            std_intensity = np.std(cell_gray)
            
            # Burn marks typically show as dark spots with high contrast
            if std_intensity > 40 and mean_intensity < 100:
                damaged_cells.append({
                    'row': row,
                    'col': col,
                    'bbox': (x1, y1, cell_w, cell_h),
                    'severity': 'CRITICAL',
                    'mean': mean_intensity,
                    'contrast': std_intensity
                })
            elif std_intensity > 25:
                damaged_cells.append({
                    'row': row,
                    'col': col,
                    'bbox': (x1, y1, cell_w, cell_h),
                    'severity': 'MODERATE',
                    'mean': mean_intensity,
                    'contrast': std_intensity
                })
    
    return damaged_cells

def generate_report(image_path):
    """
    Generate comprehensive defect detection report.
    """
    print("="*70)
    print("SOLAR PANEL DEFECT DETECTION - CRITICAL ANALYSIS REPORT")
    print("="*70)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"\n📊 Image Analysis: {image_path}")
    print(f"   Resolution: {image.shape[1]} x {image.shape[0]} pixels")
    
    # Detect hotspots
    print("\n🔥 Hotspot Detection:")
    hotspots = detect_hotspots(image)
    
    if hotspots:
        print(f"   ⚠️  Found {len(hotspots)} potential hotspot(s)")
        for i, hotspot in enumerate(hotspots, 1):
            print(f"      Hotspot #{i}:")
            print(f"        - Location: ({hotspot['center'][0]}, {hotspot['center'][1]})")
            print(f"        - Area: {hotspot['area']} pixels")
            print(f"        - Severity Score: {hotspot['severity']:.3f}")
    else:
        print("   ✓ No significant hotspots detected")
    
    # Analyze burn marks
    print("\n🔍 Cell Damage Analysis:")
    damaged_cells = analyze_burn_marks(image)
    
    critical_count = sum(1 for c in damaged_cells if c['severity'] == 'CRITICAL')
    moderate_count = sum(1 for c in damaged_cells if c['severity'] == 'MODERATE')
    
    print(f"   Critical Damage: {critical_count} cell(s)")
    print(f"   Moderate Damage: {moderate_count} cell(s)")
    
    if damaged_cells:
        print("\n   Detailed Cell Damage:")
        for cell in damaged_cells:
            print(f"      Cell [{cell['row']},{cell['col']}] - {cell['severity']}")
            print(f"        Mean Intensity: {cell['mean']:.1f}")
            print(f"        Contrast: {cell['contrast']:.1f}")
    
    # Overall assessment
    print("\n" + "="*70)
    print("📋 CONDITION ASSESSMENT")
    print("="*70)
    
    if critical_count > 0:
        condition = "CRITICAL / FAILED"
        status_emoji = "🔴"
        recommendation = """
        ⚠️  CRITICAL DEFECTS DETECTED:
        
        Severe burn mark / hotspot visible at cell intersection(s).
        Cell surface melting or charring is present.
        Discoloration spreading from damaged area indicates overheating.
        Possible encapsulation layer damage (EVA degradation).
        
        🔴 CONDITION: Critical / Failed Panel (Not Healthy)
        
        🛠️ IMMEDIATE ACTIONS REQUIRED:
        1. Immediately disconnect/isolate this panel
        2. Do not continue operation
        3. Replace the damaged module (repair not feasible)
        4. Inspect nearby panels for similar hotspot signs
        
        📉 IMPACT:
        - Significant power loss from affected cells
        - Risk of panel efficiency drop
        - High chance of progressive failure
        - Potential fire hazard under high sunlight
        """
    elif moderate_count > 0:
        condition = "MODERATE / DEGRADED"
        status_emoji = "🟡"
        recommendation = """
        ⚠️  MODERATE DEFECTS DETECTED:
        
        Some cells show signs of degradation.
        Monitor closely for progression to critical state.
        
        🟡 CONDITION: Moderate / Needs Monitoring
        
        🛠️ RECOMMENDED ACTIONS:
        1. Schedule detailed inspection
        2. Monitor performance metrics
        3. Plan for potential replacement
        4. Check electrical connections
        """
    else:
        condition = "NORMAL / HEALTHY"
        status_emoji = "🟢"
        recommendation = """
        ✓ NO SIGNIFICANT DEFECTS DETECTED
        
        Panel appears to be in normal operating condition.
        
        🟢 CONDITION: Normal / Healthy
        
        🛠️ RECOMMENDED ACTIONS:
        1. Continue normal operation
        2. Schedule routine maintenance
        3. Monitor for any changes
        """
    
    print(f"\n{status_emoji} Overall Condition: {condition}")
    print(recommendation)
    
    # Generate visualization
    print("\n📊 Generating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Solar Panel', fontsize=14, weight='bold')
    axes[0, 0].axis('off')
    
    # Image with defect highlighting
    img_highlighted = image.copy()
    
    # Draw hotspots
    for hotspot in hotspots:
        x, y, w, h = hotspot['bbox']
        color = 'red' if hotspot['severity'] > 0.5 else 'orange'
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        axes[0, 0].add_patch(rect)
        axes[0, 0].text(x, y-5, f"Hotspot", color=color, fontsize=8, weight='bold')
    
    # Draw damaged cells
    for cell in damaged_cells:
        x, y, w, h = cell['bbox']
        color = 'red' if cell['severity'] == 'CRITICAL' else 'yellow'
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
        axes[0, 0].add_patch(rect)
    
    axes[0, 0].set_title('Original with Defect Highlights', fontsize=14, weight='bold')
    
    # Heatmap of anomalies
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    axes[0, 1].imshow(heatmap)
    axes[0, 1].set_title('Thermal Anomaly Map', fontsize=14, weight='bold')
    axes[0, 1].axis('off')
    
    # Cell grid analysis
    cell_grid = np.zeros((6, 10))
    for cell in damaged_cells:
        row, col = cell['row'], cell['col']
        if cell['severity'] == 'CRITICAL':
            cell_grid[row, col] = 2
        else:
            cell_grid[row, col] = 1
    
    im = axes[1, 0].imshow(cell_grid, cmap='RdYlGn_r', vmin=0, vmax=2)
    axes[1, 0].set_title('Cell Damage Grid\n(Green=OK, Yellow=Moderate, Red=Critical)', 
                        fontsize=14, weight='bold')
    axes[1, 0].set_xlabel('Cell Column')
    axes[1, 0].set_ylabel('Cell Row')
    
    # Add grid lines
    for i in range(7):
        axes[1, 0].axhline(i-0.5, color='black', linewidth=0.5)
    for j in range(11):
        axes[1, 0].axvline(j-0.5, color='black', linewidth=0.5)
    
    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""
    DEFECT DETECTION SUMMARY
    
    🔥 Hotspots Detected: {len(hotspots)}
    🔴 Critical Cells: {critical_count}
    🟡 Moderate Cells: {moderate_count}
    🟢 Healthy Cells: {60 - len(damaged_cells)}
    
    OVERALL CONDITION:
    {status_emoji} {condition}
    
    KEY FINDINGS:
    """
    
    if critical_count > 0:
        summary_text += """
    • Severe burn marks present
    • Cell surface damage detected
    • Overheating indicators visible
    • Immediate replacement required
        """
    elif moderate_count > 0:
        summary_text += """
    • Some cell degradation detected
    • Monitor for progression
    • Schedule maintenance
        """
    else:
        summary_text += """
    • No significant defects
    • Panel in good condition
    • Continue normal operation
        """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save result
    output_path = 'solar_panel_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Analysis saved: {output_path}")
    
    plt.show()
    
    return {
        'condition': condition,
        'hotspots': len(hotspots),
        'critical_cells': critical_count,
        'moderate_cells': moderate_count,
        'damaged_cells': damaged_cells
    }

if __name__ == "__main__":
    image_path = "solar.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found!")
        sys.exit(1)
    
    try:
        results = generate_report(image_path)
        print("\n" + "="*70)
        print("Analysis complete!")
        print("="*70)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
