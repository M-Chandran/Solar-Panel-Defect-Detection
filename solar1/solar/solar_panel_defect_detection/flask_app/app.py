"""
Solar Panel Defect Detection - Flask Application
Complete web application with login, signup, and defect detection
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename
import os
import csv
import hashlib
import datetime
from collections import Counter
import numpy as np
from PIL import Image
import cv2
import io

app = Flask(__name__)
app.secret_key = 'solar-panel-secret-key-2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# CSV file for user data
USERS_CSV = 'users.csv'
HISTORY_CSV = 'history.csv'

# Initialize CSV files
def init_csv_files():
    # Users CSV
    if not os.path.exists(USERS_CSV):
        with open(USERS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['username', 'password', 'email', 'created_at'])
    
    # History CSV
    if not os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['username', 'filename', 'defects_found', 'condition', 'timestamp'])

init_csv_files()

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

def save_user(username, password, email):
    with open(USERS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([username, hash_password(password), email, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

def verify_user(username, password):
    with open(USERS_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row[0] == username and verify_password(password, row[1]):
                return True
    return False

def user_exists(username):
    with open(USERS_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[0] == username:
                return True
    return False

def save_history(username, filename, defects_count, condition):
    with open(HISTORY_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([username, filename, defects_count, condition, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

def get_history(username):
    history = []
    with open(HISTORY_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[0] == username:
                history.append({
                    'filename': row[1],
                    'defects_found': row[2],
                    'condition': row[3],
                    'timestamp': row[4]
                })
    return list(reversed(history))

# ==================== DEFECT DETECTION ====================

class EnhancedDefectDetector:
    """Enhanced defect detection with multiple methods"""
    
    def __init__(self, sensitivity=0.5):
        self.sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
        self.rows = 6
        self.cols = 10
        self.defect_priority = {
            'Burn Mark': 5,
            'Thermal Anomaly': 4,
            'Hotspot': 3,
            'Crack': 2,
            'Discoloration': 1
        }

    def _build_grid_mask(self, gray):
        """Detect dominant panel grid lines for crack false-positive suppression."""
        h, w = gray.shape[:2]
        smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
        )
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 15), 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 15)))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        grid_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        return cv2.dilate(grid_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    
    def detect_all_methods(self, image_array):
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
        hsv, lab = None, None
        if len(image_array.shape) == 3:
            hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        edges = cv2.Canny(gray, 40, 120)
        grid_mask = self._build_grid_mask(gray)

        method1 = self.method_statistics_based(image_array, gray=gray)
        method2 = self.method_edge_based(image_array, gray=gray, edges=edges, grid_mask=grid_mask)
        method3 = self.method_color_based(image_array, hsv=hsv, lab=lab, gray=gray, edges=edges, grid_mask=grid_mask)
        method4 = self.method_thermal_based(image_array, gray=gray)

        results = {
            'method1_statistics': method1,
            'method2_edge': method2,
            'method3_color': method3,
            'method4_thermal': method4,
            'combined': self.combine_results(method_results=[method1, method2, method3, method4])
        }
        return results
    
    def method_statistics_based(self, image, gray=None):
        if gray is None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        rows, cols = self.rows, self.cols
        cell_h, cell_w = h // rows, w // cols
        global_mean = float(np.mean(gray))
        global_std = float(np.std(gray) + 1e-8)
        burn_mean_factor = 0.82 + (1.0 - self.sensitivity) * 0.08
        burn_std_factor = 0.55 + (1.0 - self.sensitivity) * 0.25
        hotspot_mean_factor = 1.18 - self.sensitivity * 0.08
        hotspot_std_factor = 0.40 + (1.0 - self.sensitivity) * 0.25
        
        defects = []
        for row in range(rows):
            for col in range(cols):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                cell = gray[y1:y2, x1:x2]
                mean_val = float(np.mean(cell))
                std_val = float(np.std(cell))
                min_val = float(np.min(cell))
                max_val = float(np.max(cell))
                
                if (mean_val < global_mean * burn_mean_factor and
                    std_val > global_std * burn_std_factor and
                    min_val < global_mean * 0.60):
                    darkness = max(0.0, (global_mean * burn_mean_factor - mean_val) / max(global_mean, 1e-8))
                    texture = max(0.0, std_val / max(global_std, 1e-8))
                    severity = float(np.clip(0.55 * darkness + 0.45 * (texture / 2.2), 0.0, 1.0))
                    if severity > (0.38 + (1.0 - self.sensitivity) * 0.20):
                        defects.append({
                            'type': 'Burn Mark', 'row': row, 'col': col,
                            'severity': severity, 'confidence': severity,
                            'bbox': (x1, y1, x2, y2), 'method': 'statistical'
                        })
                elif (mean_val > global_mean * hotspot_mean_factor and
                      std_val > global_std * hotspot_std_factor and
                      max_val > min(255.0, global_mean * 1.30)):
                    brightness = max(0.0, (mean_val - global_mean * hotspot_mean_factor) / max(global_mean, 1e-8))
                    texture = max(0.0, std_val / max(global_std, 1e-8))
                    peak = max(0.0, (max_val - 235.0) / 20.0)
                    severity = float(np.clip(0.45 * brightness + 0.35 * (texture / 2.0) + 0.20 * peak, 0.0, 1.0))
                    if severity > (0.34 + (1.0 - self.sensitivity) * 0.18):
                        defects.append({
                            'type': 'Hotspot', 'row': row, 'col': col,
                            'severity': severity, 'confidence': severity,
                            'bbox': (x1, y1, x2, y2), 'method': 'statistical'
                        })
        return defects
    
    def method_edge_based(self, image, gray=None, edges=None, grid_mask=None):
        if gray is None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        if edges is None:
            edges = cv2.Canny(gray, 40, 120)
        if grid_mask is None:
            grid_mask = self._build_grid_mask(gray)
        h, w = gray.shape
        rows, cols = self.rows, self.cols
        cell_h, cell_w = h // rows, w // cols
        global_mean = float(np.mean(gray))
        edge_thresh = 0.04 + (1.0 - self.sensitivity) * 0.03
        severity_thresh = 0.20 + (1.0 - self.sensitivity) * 0.16
        
        defects = []
        for row in range(rows):
            for col in range(cols):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                cell_edges = edges[y1:y2, x1:x2]
                cell_gray = gray[y1:y2, x1:x2]
                grid_cell = grid_mask[y1:y2, x1:x2]
                edge_ratio = float(np.sum(cell_edges > 0) / cell_edges.size) if cell_edges.size else 0.0
                grid_ratio = float(np.sum(grid_cell > 0) / grid_cell.size) if grid_cell.size else 0.0
                mean_gray = float(np.mean(cell_gray))
                if mean_gray < global_mean * 0.72:
                    continue
                if edge_ratio <= edge_thresh or grid_ratio >= 0.45:
                    continue

                min_dim = max(8, min(cell_h, cell_w))
                lines = cv2.HoughLinesP(
                    cell_edges,
                    1,
                    np.pi / 180,
                    threshold=max(6, int(0.12 * min_dim)),
                    minLineLength=max(6, int(0.25 * min_dim)),
                    maxLineGap=max(2, int(0.08 * min_dim))
                )
                line_density = 0.0
                if lines is not None:
                    total_length = 0.0
                    for line in lines[:, 0]:
                        x_a, y_a, x_b, y_b = line
                        total_length += float(np.hypot(x_b - x_a, y_b - y_a))
                    line_density = total_length / max(1.0, float(cell_h + cell_w))

                if line_density > 0.18:
                    strength = max(0.0, (edge_ratio - edge_thresh) / max(edge_thresh, 1e-8))
                    severity = float(np.clip(0.45 * strength + 0.40 * line_density + 0.15 * (1.0 - grid_ratio), 0.0, 1.0))
                    if severity > severity_thresh:
                        defects.append({
                            'type': 'Crack', 'row': row, 'col': col,
                            'severity': severity, 'confidence': severity,
                            'bbox': (x1, y1, x2, y2), 'method': 'edge'
                        })
        return defects
    
    def method_color_based(self, image, hsv=None, lab=None, gray=None, edges=None, grid_mask=None):
        if len(image.shape) != 3:
            return []
        if gray is None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if hsv is None:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if lab is None:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        if edges is None:
            edges = cv2.Canny(gray, 40, 120)
        if grid_mask is None:
            grid_mask = self._build_grid_mask(gray)
        rows, cols = self.rows, self.cols
        cell_h, cell_w = image.shape[0] // rows, image.shape[1] // cols
        global_a_mean = float(np.mean(lab[:, :, 1]))
        global_b_mean = float(np.mean(lab[:, :, 2]))
        color_thresh = 14.0 + (1.0 - self.sensitivity) * 8.0
        hue_thresh = 12.0 + (1.0 - self.sensitivity) * 8.0
        edge_limit = 0.24 + (1.0 - self.sensitivity) * 0.08
        severity_thresh = 0.22 + (1.0 - self.sensitivity) * 0.18
        
        defects = []
        for row in range(rows):
            for col in range(cols):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                cell_lab = lab[y1:y2, x1:x2]
                cell_hsv = hsv[y1:y2, x1:x2]
                cell_gray = gray[y1:y2, x1:x2]
                cell_edges = edges[y1:y2, x1:x2]
                cell_grid = grid_mask[y1:y2, x1:x2]

                color_var = float(np.std(cell_lab[:, :, 1]) + np.std(cell_lab[:, :, 2]))
                hue_var = float(np.std(cell_hsv[:, :, 0]))
                mean_gray = float(np.mean(cell_gray))
                edge_ratio = float(np.sum(cell_edges > 0) / cell_edges.size) if cell_edges.size else 0.0
                grid_ratio = float(np.sum(cell_grid > 0) / cell_grid.size) if cell_grid.size else 0.0
                chromatic_shift = abs(float(np.mean(cell_lab[:, :, 1])) - global_a_mean) + abs(float(np.mean(cell_lab[:, :, 2])) - global_b_mean)

                if mean_gray < 55.0 or mean_gray > 235.0 or edge_ratio >= edge_limit or grid_ratio > 0.55:
                    continue

                if color_var > color_thresh or (hue_var > hue_thresh and chromatic_shift > 6.0):
                    color_strength = max(0.0, (color_var - color_thresh) / max(color_thresh, 1e-8))
                    hue_strength = max(0.0, (hue_var - hue_thresh) / max(hue_thresh, 1e-8))
                    shift_strength = min(1.0, chromatic_shift / 20.0)
                    severity = float(np.clip(0.40 * color_strength + 0.30 * hue_strength + 0.30 * shift_strength + 0.20, 0.0, 1.0))
                    if severity > severity_thresh:
                        defects.append({
                            'type': 'Discoloration', 'row': row, 'col': col,
                            'severity': severity, 'confidence': severity,
                            'bbox': (x1, y1, x2, y2), 'method': 'color'
                        })
        return defects
    
    def method_thermal_based(self, image, gray=None):
        if gray is None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        thermal_thresh = int(max(205, min(235, 220 - self.sensitivity * 12)))
        _, bright = cv2.threshold(gray, thermal_thresh, 255, cv2.THRESH_BINARY)
        rows, cols = self.rows, self.cols
        cell_h, cell_w = gray.shape[0] // rows, gray.shape[1] // cols
        bright_ratio_thresh = 0.20 + (1.0 - self.sensitivity) * 0.12
        severity_thresh = 0.32 + (1.0 - self.sensitivity) * 0.20
        
        defects = []
        for row in range(rows):
            for col in range(cols):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                cell = bright[y1:y2, x1:x2]
                cell_gray = gray[y1:y2, x1:x2]
                bright_ratio = float(np.sum(cell > 0) / cell.size) if cell.size else 0.0
                
                if bright_ratio > bright_ratio_thresh:
                    local_mean = float(np.mean(cell_gray))
                    local_max = float(np.max(cell_gray))
                    ratio_strength = max(0.0, bright_ratio / max(bright_ratio_thresh, 1e-8) - 1.0)
                    peak_strength = max(0.0, (local_max - 235.0) / 20.0)
                    mean_strength = max(0.0, (local_mean - 180.0) / 60.0)
                    severity = float(np.clip(0.55 * ratio_strength + 0.25 * peak_strength + 0.20 * mean_strength, 0.0, 1.0))
                    if severity > severity_thresh:
                        defects.append({
                            'type': 'Thermal Anomaly', 'row': row, 'col': col,
                            'severity': severity, 'confidence': severity,
                            'bbox': (x1, y1, x2, y2), 'method': 'thermal'
                        })
        return defects
    
    def combine_results(self, image=None, method_results=None):
        all_defects = []
        if method_results is None:
            if image is None:
                return []
            all_defects.extend(self.method_statistics_based(image))
            all_defects.extend(self.method_edge_based(image))
            all_defects.extend(self.method_color_based(image))
            all_defects.extend(self.method_thermal_based(image))
        else:
            for defects in method_results:
                all_defects.extend(defects)

        # Keep strongest candidate for each (row, col, type).
        deduped = {}
        for defect in all_defects:
            key = (defect['row'], defect['col'], defect['type'])
            current = deduped.get(key)
            if current is None or defect.get('severity', 0.0) > current.get('severity', 0.0):
                deduped[key] = defect

        # Choose one final label per cell so categories are mutually exclusive.
        best_by_cell = {}
        for defect in deduped.values():
            cell_key = (defect['row'], defect['col'])
            current = best_by_cell.get(cell_key)
            if current is None:
                best_by_cell[cell_key] = defect
                continue

            cur_sev = float(defect.get('severity', 0.0))
            old_sev = float(current.get('severity', 0.0))
            cur_pri = self.defect_priority.get(defect.get('type'), 0)
            old_pri = self.defect_priority.get(current.get('type'), 0)
            cur_score = cur_sev + 0.12 * (cur_pri / 5.0)
            old_score = old_sev + 0.12 * (old_pri / 5.0)

            if cur_score > old_score:
                best_by_cell[cell_key] = defect
                continue

            if abs(cur_score - old_score) <= 1e-8 and float(defect.get('confidence', 0.0)) > float(current.get('confidence', 0.0)):
                best_by_cell[cell_key] = defect

        return sorted(best_by_cell.values(), key=lambda d: (d['row'], d['col']))


def generate_visualizations(image_array, defects, method_results):
    """Generate all visualization steps"""
    visualizations = {}
    h, w = image_array.shape[:2]
    rows, cols = 6, 10
    cell_h, cell_w = h // rows, w // cols
    
    # 1. Original
    visualizations['original'] = image_array.copy()
    
    # 2. Grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
    visualizations['grayscale'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # 3. Grid
    grid_img = image_array.copy()
    for row in range(rows + 1):
        cv2.line(grid_img, (0, row * cell_h), (w, row * cell_h), (255, 255, 255), 1)
    for col in range(cols + 1):
        cv2.line(grid_img, (col * cell_w, 0), (col * cell_w, h), (255, 255, 255), 1)
    visualizations['grid'] = grid_img
    
    # 4. Edges
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored[edges > 0] = [0, 255, 255]
    visualizations['edges'] = edges_colored
    
    # 5. Heatmap
    global_mean = np.mean(gray)
    heatmap = np.zeros((h, w), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cell = gray[y1:y2, x1:x2]
            deviation = abs(np.mean(cell) - global_mean) / (np.std(gray) + 1e-8)
            heatmap[y1:y2, x1:x2] = deviation
    
    heatmap_norm = (heatmap / (heatmap.max() + 1e-8) * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    visualizations['heatmap'] = heatmap_colored
    
    # 6. Method Results
    method_viz = {}
    for method_name, method_defects in method_results.items():
        if method_name == 'combined':
            continue
        method_img = image_array.copy()
        colors = {'method1_statistics': (255, 0, 0), 'method2_edge': (0, 255, 255), 
                  'method3_color': (255, 255, 0), 'method4_thermal': (255, 0, 255)}
        for d in method_defects:
            x1, y1, x2, y2 = d['bbox']
            cv2.rectangle(method_img, (x1, y1), (x2, y2), colors.get(method_name, (255, 255, 255)), 2)
        method_viz[method_name] = method_img
    visualizations['methods'] = method_viz
    
    # 7. Final
    final = image_array.copy()
    color_map = {
        'Burn Mark': (0, 0, 255),        # Red
        'Hotspot': (255, 0, 0),          # Blue
        'Crack': (0, 255, 255),          # Yellow
        'Discoloration': (255, 255, 0),  # Cyan
        'Thermal Anomaly': (255, 0, 255) # Magenta
    }
    safe_color = (0, 255, 0)  # Green

    # Highlight safe (non-defective) cells first.
    defect_cells = {(d.get('row'), d.get('col')) for d in defects}
    safe_overlay = final.copy()
    for row in range(rows):
        for col in range(cols):
            if (row, col) in defect_cells:
                continue
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cv2.rectangle(safe_overlay, (x1, y1), (x2, y2), safe_color, -1)

    final = cv2.addWeighted(final, 0.88, safe_overlay, 0.12, 0)
    for row in range(rows):
        for col in range(cols):
            if (row, col) in defect_cells:
                continue
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cv2.rectangle(final, (x1, y1), (x2, y2), safe_color, 1)

    for defect in defects:
        x1, y1, x2, y2 = defect['bbox']
        color = color_map.get(defect['type'], (255, 255, 255))
        
        overlay = final.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        final = cv2.addWeighted(final, 0.7, overlay, 0.3, 0)
        
        cv2.rectangle(final, (x1, y1), (x2, y2), color, 3)
        cv2.putText(final, defect['type'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    visualizations['final'] = final
    
    return visualizations


def save_image_to_bytes(img):
    """Convert numpy array to bytes for display"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return buf


# ==================== ROUTES ====================

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if verify_user(username, password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        if user_exists(username):
            flash('Username already exists', 'error')
        else:
            save_user(username, password, email)
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
    
    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    history = get_history(username)
    
    return render_template('dashboard.html', username=username, history=history[:10])


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    sensitivity = 0.5
    if request.method == 'POST':
        sensitivity = float(request.form.get('sensitivity', 0.5))
    
    return render_template('detect.html', sensitivity=sensitivity, username=username)


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('detect'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('detect'))
    
    if file and allowed_file(file.filename):
        # Read and process image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        image_array = np.array(img)
        
        # Run detection
        detector = EnhancedDefectDetector(sensitivity=0.5)
        method_results = detector.detect_all_methods(image_array)
        defects = method_results['combined']

        # Build compact findings summary and safe-panel stats
        total_cells = detector.rows * detector.cols
        defect_cells = {(d['row'], d['col']) for d in defects}
        defective_cell_count = len(defect_cells)
        safe_panel_count = max(0, total_cells - defective_cell_count)
        safe_panel_percent = (safe_panel_count / total_cells * 100.0) if total_cells else 0.0

        type_counter = Counter(d['type'] for d in defects)
        type_conf = {}
        for d in defects:
            type_conf.setdefault(d['type'], []).append(d['confidence'])

        defect_summary = []
        for defect_type, count in type_counter.most_common():
            confidences = type_conf.get(defect_type, [])
            avg_confidence = float(np.mean(confidences)) if confidences else 0.0
            defect_summary.append({
                'type': defect_type,
                'count': int(count),
                'avg_confidence': avg_confidence
            })
        
        # Generate visualizations
        visualizations = generate_visualizations(image_array, defects, method_results)
        
        # Save history
        condition = "Critical" if defective_cell_count > 5 else "Degraded" if defective_cell_count > 0 else "Normal"
        save_history(session['username'], file.filename, defective_cell_count, condition)
        
        # Convert images to base64 for display
        import base64
        
        def img_to_base64(img):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
            pil_img = Image.fromarray(img_rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode()
        
        # Get method images
        method_images = {}
        for name, img in visualizations['methods'].items():
            method_images[name] = img_to_base64(img)
        
        return render_template('result.html',
                           username=session['username'],
                           filename=file.filename,
                           defects=defects,
                           defect_summary=defect_summary,
                           defect_count=defective_cell_count,
                           total_cells=total_cells,
                           defective_cell_count=defective_cell_count,
                           safe_panel_count=safe_panel_count,
                           safe_panel_percent=safe_panel_percent,
                           condition=condition,
                           original=img_to_base64(visualizations['original']),
                           grayscale=img_to_base64(visualizations['grayscale']),
                           grid=img_to_base64(visualizations['grid']),
                           edges=img_to_base64(visualizations['edges']),
                           heatmap=img_to_base64(visualizations['heatmap']),
                           final=img_to_base64(visualizations['final']),
                           method_images=method_images)
    
    return redirect(url_for('detect'))


@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    history = get_history(username)
    
    return render_template('history.html', username=username, history=history)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
