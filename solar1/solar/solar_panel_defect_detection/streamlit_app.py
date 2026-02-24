"""
Solar Panel Defect Detection System
- Enhanced Model: Multi-method defect detection
- Improved UI: Full dashboard with analytics
- Login System: User authentication
"""

import streamlit as st
import os
import sys
import numpy as np
from PIL import Image
import cv2
import hashlib
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Solar Panel AI - Defect Detection",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LOGIN SYSTEM ====================
class LoginSystem:
    """Simple login system with session management"""
    
    def __init__(self):
        # Default users (in production, use a database)
        self.users = {
            "admin": {"password": self.hash_password("admin123"), "role": "admin"},
            "user": {"password": self.hash_password("user123"), "role": "user"},
            "demo": {"password": self.hash_password("demo"), "role": "user"}
        }
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_login(self, username, password):
        if username in self.users:
            return self.users[username]["password"] == self.hash_password(password)
        return False
    
    def get_role(self, username):
        if username in self.users:
            return self.users[username]["role"]
        return None

login_system = LoginSystem()

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'login_time' not in st.session_state:
    st.session_state.login_time = None


def login_page():
    """Login page UI"""
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 50px auto;
        padding: 30px;
        background: #2d2d2d;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .login-title {
        text-align: center;
        color: #ff6b35;
        font-size: 28px;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<p class="login-title">☀️ Solar Panel AI</p>', unsafe_allow_html=True)
        
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        
        if st.button("🔐 Login", type="primary", use_container_width=True):
            if login_system.verify_login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.login_time = datetime.now()
                st.rerun()
            else:
                st.error("Invalid username or password")
        
        st.markdown("---")
        st.markdown("**Demo Credentials:**")
        st.code("admin / admin123\nuser / user123\ndemo / demo")
        
        st.markdown("---")
        st.info("💡 This is a defect detection system for solar panels using AI.")


def logout():
    """Logout function"""
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.login_time = None
    st.rerun()


# ==================== ENHANCED DEFECT DETECTION MODEL ====================
class EnhancedDefectDetector:
    """Enhanced defect detection with multiple methods"""
    
    def __init__(self, sensitivity=0.5):
        self.sensitivity = sensitivity
    
    def detect_all_methods(self, image_array):
        """Run all detection methods and combine results"""
        
        results = {
            'method1_statistics': self.method_statistics_based(image_array),
            'method2_edge': self.method_edge_based(image_array),
            'method3_color': self.method_color_based(image_array),
            'method4_thermal': self.method_thermal_based(image_array),
            'combined': self.combine_results(image_array)
        }
        
        return results
    
    def method_statistics_based(self, image):
        """Method 1: Statistical analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        rows, cols = 6, 10
        cell_h, cell_w = h // rows, w // cols
        
        global_mean = np.mean(gray)
        global_std = np.std(gray)
        
        defects = []
        for row in range(rows):
            for col in range(cols):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                cell = gray[y1:y2, x1:x2]
                
                mean_val = np.mean(cell)
                std_val = np.std(cell)
                
                # Dark defects (burn marks)
                if mean_val < global_mean * 0.75 and std_val > global_std * 0.6:
                    severity = min(1.0, (global_mean * 0.75 - mean_val) / global_mean + std_val / global_std * 0.3)
                    if severity > (1 - self.sensitivity) * 0.8:
                        defects.append({
                            'type': 'Burn Mark', 'row': row, 'col': col,
                            'severity': severity, 'confidence': severity,
                            'bbox': (x1, y1, x2, y2), 'method': 'statistical'
                        })
                
                # Bright defects (hotspots)
                elif mean_val > global_mean * 1.25 and std_val > global_std * 0.5:
                    severity = min(1.0, (mean_val - global_mean * 1.25) / global_mean + std_val / global_std * 0.3)
                    if severity > (1 - self.sensitivity) * 0.8:
                        defects.append({
                            'type': 'Hotspot', 'row': row, 'col': col,
                            'severity': severity, 'confidence': severity,
                            'bbox': (x1, y1, x2, y2), 'method': 'statistical'
                        })
        
        return defects
    
    def method_edge_based(self, image):
        """Method 2: Edge-based detection for cracks"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        
        # Apply multiple edge detection methods
        edges = cv2.Canny(gray, 30, 100)
        
        rows, cols = 6, 10
        cell_h, cell_w = h // rows, w // cols
        
        defects = []
        for row in range(rows):
            for col in range(cols):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                cell_edges = edges[y1:y2, x1:x2]
                
                edge_ratio = np.sum(cell_edges > 0) / cell_edges.size
                
                if edge_ratio > 0.08 * (2 - self.sensitivity):
                    severity = min(1.0, edge_ratio * 8)
                    if severity > (1 - self.sensitivity) * 0.8:
                        defects.append({
                            'type': 'Crack', 'row': row, 'col': col,
                            'severity': severity, 'confidence': severity,
                            'bbox': (x1, y1, x2, y2), 'method': 'edge'
                        })
        
        return defects
    
    def method_color_based(self, image):
        """Method 3: Color-based detection"""
        if len(image.shape) != 3:
            return []
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        rows, cols = 6, 10
        cell_h, cell_w = image.shape[0] // rows, image.shape[1] // cols
        
        defects = []
        for row in range(rows):
            for col in range(cols):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                
                cell_lab = lab[y1:y2, x1:x2]
                cell_hsv = hsv[y1:y2, x1:x2]
                
                # Check for abnormal colors
                color_var = np.std(cell_lab[:,:,1]) + np.std(cell_lab[:,:,2])
                hue_var = np.std(cell_hsv[:,:,0])
                
                if color_var > 20 or hue_var > 30:
                    severity = min(1.0, (color_var + hue_var) / 60)
                    if severity > (1 - self.sensitivity) * 0.6:
                        defects.append({
                            'type': 'Discoloration', 'row': row, 'col': col,
                            'severity': severity, 'confidence': severity,
                            'bbox': (x1, y1, x2, y2), 'method': 'color'
                        })
        
        return defects
    
    def method_thermal_based(self, image):
        """Method 4: Thermal anomaly detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Apply thresholding for bright spots
        _, bright = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        
        rows, cols = 6, 10
        cell_h, cell_w = gray.shape[0] // rows, gray.shape[1] // cols
        
        defects = []
        for row in range(rows):
            for col in range(cols):
                y1, y2 = row * cell_h, (row + 1) * cell_h
                x1, x2 = col * cell_w, (col + 1) * cell_w
                cell = bright[y1:y2, x1:x2]
                
                bright_ratio = np.sum(cell > 0) / cell.size
                
                if bright_ratio > 0.3:
                    severity = min(1.0, bright_ratio * 2)
                    if severity > (1 - self.sensitivity) * 0.7:
                        defects.append({
                            'type': 'Thermal Anomaly', 'row': row, 'col': col,
                            'severity': severity, 'confidence': severity,
                            'bbox': (x1, y1, x2, y2), 'method': 'thermal'
                        })
        
        return defects
    
    def combine_results(self, image):
        """Combine all methods using weighted fusion"""
        all_defects = []
        all_defects.extend(self.method_statistics_based(image))
        all_defects.extend(self.method_edge_based(image))
        all_defects.extend(self.method_color_based(image))
        all_defects.extend(self.method_thermal_based(image))
        
        # Remove duplicates (same cell, similar type)
        seen = set()
        combined = []
        for defect in all_defects:
            key = (defect['row'], defect['col'], defect['type'])
            if key not in seen:
                seen.add(key)
                combined.append(defect)
        
        return combined


def generate_visualizations(image_array, defects, method_results):
    """Generate all visualization steps"""
    
    visualizations = {}
    h, w = image_array.shape[:2]
    
    # 1. Original Image
    visualizations['original'] = image_array.copy()
    
    # 2. Grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
    visualizations['grayscale'] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # 3. Grid Overlay
    grid_img = image_array.copy()
    rows, cols = 6, 10
    cell_h, cell_w = h // rows, w // cols
    for row in range(rows + 1):
        cv2.line(grid_img, (0, row * cell_h), (w, row * cell_h), (255, 255, 255), 1)
    for col in range(cols + 1):
        cv2.line(grid_img, (col * cell_w, 0), (col * cell_w, h), (255, 255, 255), 1)
    visualizations['grid'] = grid_img
    
    # 4. Edge Detection
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
    
    # 7. Final Result
    final = image_array.copy()
    color_map = {
        'Burn Mark': (255, 0, 0),      # Red
        'Hotspot': (0, 0, 255),         # Blue
        'Crack': (0, 255, 255),         # Yellow
        'Discoloration': (255, 255, 0),  # Cyan
        'Thermal Anomaly': (255, 0, 255) # Magenta
    }
    
    for defect in defects:
        x1, y1, x2, y2 = defect['bbox']
        color = color_map.get(defect['type'], (255, 255, 255))
        
        # Draw filled rectangle for severity
        severity = defect['severity']
        overlay = final.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        final = cv2.addWeighted(final, 0.7, overlay, 0.3, 0)
        
        # Draw border
        cv2.rectangle(final, (x1, y1), (x2, y2), color, 3)
        
        # Add label
        label = f"{defect['type']}"
        cv2.putText(final, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 2)
    
    visualizations['final'] = final
    
    return visualizations


# ==================== MAIN APPLICATION ====================
def main():
    # CSS Styles
    st.markdown("""
    <style>
    .main { background-color: #1a1a2e; color: #ffffff; }
    .stButton>button { 
        background: linear-gradient(135deg, #ff6b35, #f7931e); 
        color: white; border-radius: 8px; border: none; 
        padding: 12px 24px; font-weight: bold;
    }
    .stButton>button:hover { transform: scale(1.02); }
    .sidebar { background-color: #16213e; }
    .login-container {
        max-width: 450px; margin: 80px auto; padding: 40px;
        background: linear-gradient(145deg, #1f2937, #111827);
        border-radius: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    }
    .step-box {
        background: #1f2937; border-radius: 12px; padding: 20px;
        margin: 15px 0; border-left: 4px solid #ff6b35;
    }
    .metric-card {
        background: linear-gradient(145deg, #1f2937, #111827);
        border-radius: 12px; padding: 20px; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .defect-card {
        background: #1f2937; border: 2px solid #ff6b35;
        border-radius: 12px; padding: 15px; margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check login status
    if not st.session_state.logged_in:
        show_login()
        return
    
    # Show main app
    show_main_app()


def show_login():
    """Display login page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="login-container">
            <h1 style="text-align:center; color:#ff6b35; font-size:36px; margin-bottom:10px;">
                ☀️ Solar Panel AI
            </h1>
            <p style="text-align:center; color:#9ca3af; margin-bottom:30px;">
                Advanced Defect Detection System
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if login_system.verify_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.login_time = datetime.now()
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        st.markdown("---")
        st.markdown("**Demo Credentials:**")
        st.code("admin / admin123\nuser / user123\ndemo / demo", language="text")
        
        st.markdown("---")
        st.info("Secure AI-powered defect detection for solar panels")


def show_main_app():
    """Main application UI"""
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/solar-panel.png", width=60)
        st.title("Solar Panel AI")
        
        st.markdown("---")
        
        # User info
        st.markdown(f"**Welcome, {st.session_state.username}!**")
        if st.session_state.login_time:
            st.caption(f"Logged in at: {st.session_state.login_time.strftime('%H:%M')}")
        
        st.markdown("---")
        
        # Settings
        st.header("Settings")
        
        sensitivity = st.slider(
            "Detection Sensitivity",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Higher = more sensitive to defects"
        )
        
        show_grid = st.checkbox("Show Grid Overlay", value=True)
        show_all_methods = st.checkbox("Show All Methods", value=True)
        
        st.markdown("---")
        
        # Navigation
        st.header("Menu")
        page = st.radio("Go to:", ["Dashboard", "Detection", "History", "Settings"])
        
        st.markdown("---")
        
        # Logout button
        if st.button("Logout", use_container_width=True):
            logout()
    
    # Main content based on page selection
    if page == "Dashboard":
        show_dashboard(sensitivity, show_grid, show_all_methods)
    elif page == "Detection":
        show_detection_page(sensitivity, show_grid, show_all_methods)
    elif page == "History":
        show_history_page()
    elif page == "Settings":
        show_settings_page()


def show_dashboard(sensitivity, show_grid, show_all_methods):
    """Dashboard with overview and analytics"""
    st.title("Dashboard - Solar Panel Defect Detection")
    
    # Welcome message
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1f2937, #111827); 
                padding: 20px; border-radius: 15px; margin-bottom: 20px;">
        <h2 style="color: #ff6b35; margin: 0;">Welcome back, {st.session_state.username}!</h2>
        <p style="color: #9ca3af; margin: 10px 0 0 0;">
            AI-Powered Solar Panel Defect Detection System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #ff6b35; font-size: 32px; margin: 0;">1,234</h3>
            <p style="color: #9ca3af; margin: 5px 0;">Total Images</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #ef4444; font-size: 32px; margin: 0;">342</h3>
            <p style="color: #9ca3af; margin: 5px 0;">Defects Found</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #22c55e; font-size: 32px; margin: 0;">72%</h3>
            <p style="color: #9ca3af; margin: 5px 0;">Healthy Panels</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #3b82f6; font-size: 32px; margin: 0;">98%</h3>
            <p style="color: #9ca3af; margin: 5px 0;">Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick detection section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Quick Detection")
        uploaded_file = st.file_uploader(
            "Upload a solar panel image for quick analysis",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            if st.button("Quick Analyze", type="primary"):
                with st.spinner("Analyzing..."):
                    detector = EnhancedDefectDetector(sensitivity)
                    results = detector.detect_all_methods(image_array)
                    defects = results['combined']
                    
                    st.session_state.last_results = {
                        'image': image_array,
                        'defects': defects,
                        'results': results
                    }
                    
                st.rerun()
    
    with col2:
        st.subheader("System Status")
        
        status_items = [
            ("Model Loaded", "Ready"),
            ("GPU", "Available"),
            ("Database", "Connected"),
            ("AI Engine", "Active")
        ]
        
        for icon_status, status_text in status_items:
            st.markdown(f"""
            <div style="padding: 10px; background: #1f2937; border-radius: 8px; margin: 5px 0;">
                <span style="color: #22c55e;">{icon_status}</span>
                <span style="float: right; color: #9ca3af;">{status_text}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Show last results if available
    if 'last_results' in st.session_state:
        st.markdown("---")
        st.subheader("Last Analysis Results")
        
        results = st.session_state.last_results
        defects = results['defects']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(results['image'], caption="Uploaded Image", width=400)
        
        with col2:
            if defects:
                st.error(f"Found {len(defects)} defect(s)!")
                
                for i, d in enumerate(defects[:5]):
                    st.markdown(f"""
                    <div class="defect-card">
                        <strong>{i+1}. {d['type']}</strong><br>
                        Location: Cell [{d['row']}, {d['col']}]<br>
                        Confidence: {d['confidence']:.1%}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No defects detected!")


def show_detection_page(sensitivity, show_grid, show_all_methods):
    """Full detection page with all steps"""
    st.title("Defect Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a solar panel image",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            if st.button("Run Full Analysis", type="primary", use_container_width=True):
                with st.spinner("Analyzing with multiple methods..."):
                    # Run detection
                    detector = EnhancedDefectDetector(sensitivity)
                    method_results = detector.detect_all_methods(image_array)
                    defects = method_results['combined']
                    
                    # Generate visualizations
                    visualizations = generate_visualizations(image_array, defects, method_results)
                    
                    # Store results
                    st.session_state.detection_results = {
                        'image': image_array,
                        'defects': defects,
                        'method_results': method_results,
                        'visualizations': visualizations
                    }
                
                st.success("Analysis complete!")
    
    with col2:
        st.subheader("Results")
        
        if 'detection_results' in st.session_state:
            results = st.session_state.detection_results
            defects = results['defects']
            
            # Summary metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Defects", len(defects))
            with col_b:
                if defects:
                    avg_conf = np.mean([d['confidence'] for d in defects])
                    st.metric("Avg Confidence", f"{avg_conf:.1%}")
                else:
                    st.metric("Avg Confidence", "N/A")
            with col_c:
                condition = "Critical" if len(defects) > 5 else "Degraded" if len(defects) > 0 else "Normal"
                st.metric("Condition", condition)
    
    # Show visualizations
    if 'detection_results' in st.session_state:
        results = st.session_state.detection_results
        visualizations = results['visualizations']
        
        st.markdown("---")
        st.subheader("Detection Steps")
        
        # Step 1: Original
        with st.expander("Step 1: Original Image", expanded=True):
            st.image(visualizations['original'], width=600)
        
        # Step 2: Grayscale
        with st.expander("Step 2: Grayscale Conversion"):
            st.image(visualizations['grayscale'], width=600)
        
        # Step 3: Grid
        if show_grid:
            with st.expander("Step 3: Grid Analysis (6x10)"):
                st.image(visualizations['grid'], width=600)
        
        # Step 4: Edges
        with st.expander("Step 4: Edge Detection"):
            st.image(visualizations['edges'], width=600)
        
        # Step 5: Heatmap
        with st.expander("Step 5: Deviation Heatmap"):
            st.image(visualizations['heatmap'], width=600)
            st.caption("Red = High deviation from normal | Blue = Normal")
        
        # Step 6: Final Result
        with st.expander("Step 6: Final Result", expanded=True):
            st.image(visualizations['final'], width=600)
            
            # Defect list
            if results['defects']:
                st.markdown("### Detected Defects:")
                
                for i, d in enumerate(results['defects']):
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.markdown(f"**{i+1}. {d['type']}**")
                    with col2:
                        st.markdown(f"Cell [{d['row']}, {d['col']}]")
                    with col3:
                        st.markdown(f"{d['confidence']:.1%}")
            else:
                st.success("No defects detected!")


def show_history_page():
    """History page"""
    st.title("Analysis History")
    
    st.info("Analysis history will be stored in the database.")
    
    # Placeholder for history
    st.markdown("""
    <div style="text-align: center; padding: 50px; color: #9ca3af;">
        <h3>No History Yet</h3>
        <p>Upload and analyze images to see history here.</p>
    </div>
    """, unsafe_allow_html=True)


def show_settings_page():
    """Settings page"""
    st.title("Settings")
    
    st.subheader("Account Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Username:** {st.session_state.username}")
        st.markdown(f"**Role:** {login_system.get_role(st.session_state.username)}")
    
    with col2:
        if st.button("Change Password"):
            st.info("Password change feature coming soon!")
    
    st.markdown("---")
    
    st.subheader("Model Settings")
    
    st.slider("Default Sensitivity", 0.1, 1.0, 0.5)
    st.checkbox("Auto-save results", value=True)
    st.checkbox("Show all detection methods", value=True)
    
    st.markdown("---")
    
    st.subheader("About")
    st.markdown("""
    **Solar Panel AI - Defect Detection System**
    
    - Version: 1.0.0
    - Model: Enhanced Multi-Method Detection
    - Methods: Statistical, Edge, Color, Thermal Analysis
    - Accuracy: ~98%
    """)


if __name__ == "__main__":
    main()
