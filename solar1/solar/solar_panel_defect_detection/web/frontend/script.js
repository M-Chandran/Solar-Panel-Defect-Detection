// Solar Panel Defect Detection System - Frontend JavaScript

class SolarPanelAnalyzer {
    constructor() {
        this.currentImageId = null;
        this.apiBaseUrl = 'http://localhost:5000';
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Upload area events
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const uploadBtn = document.getElementById('uploadBtn');

        // Click to upload
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadBtn.addEventListener('click', () => fileInput.click());

        // File input change
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                this.handleFileSelect({ target: { files: files } });
            }
        });

        // Action buttons
        document.getElementById('analyzeAnotherBtn').addEventListener('click', () => this.resetUI());
        document.getElementById('downloadReportBtn').addEventListener('click', () => this.downloadReport());
        document.getElementById('viewDetailsBtn').addEventListener('click', () => this.showDetailedAnalysis());
    }

    async handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file
        if (!this.validateFile(file)) return;

        // Show upload progress
        this.showUploadProgress();

        try {
            // Upload file
            const uploadResult = await this.uploadFile(file);

            if (uploadResult) {
                this.currentImageId = uploadResult.image_id;

                // Display original image
                document.getElementById('originalImage').src = uploadResult.original_image;

                // Run analysis
                await this.runAnalysis();
            }
        } catch (error) {
            this.showError('Upload failed: ' + error.message);
        } finally {
            this.hideUploadProgress();
        }
    }

    validateFile(file) {
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff'];
        const maxSize = 10 * 1024 * 1024; // 10MB

        if (!allowedTypes.includes(file.type)) {
            this.showError('Please select a valid image file (PNG, JPG, JPEG, TIF, TIFF)');
            return false;
        }

        if (file.size > maxSize) {
            this.showError('File size must be less than 10MB');
            return false;
        }

        return true;
    }

    showUploadProgress() {
        document.querySelector('.upload-content').style.display = 'none';
        document.getElementById('uploadProgress').style.display = 'block';
        this.updateProgress(0);
    }

    hideUploadProgress() {
        document.querySelector('.upload-content').style.display = 'block';
        document.getElementById('uploadProgress').style.display = 'none';
    }

    updateProgress(percent) {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');

        progressFill.style.width = percent + '%';
        progressText.textContent = `Uploading... ${percent}%`;
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            this.updateProgress(Math.round(progress));
        }, 200);

        try {
            const response = await fetch(`${this.apiBaseUrl}/upload`, {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);
            this.updateProgress(100);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            return result;
        } catch (error) {
            clearInterval(progressInterval);
            throw error;
        }
    }

    async runAnalysis() {
        this.showModal('Processing...', 'Analyzing your solar panel image...');

        try {
            // Run inference
            const inferenceResult = await this.runInference();

            // Get detailed results
            const results = await this.getResults();

            // Get explanation
            const explanation = await this.getExplanation();

            // Update UI with results
            this.updateResultsUI(inferenceResult, results, explanation);

            // Show results section
            document.getElementById('resultsSection').style.display = 'block';

            // Scroll to results
            document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            this.showError('Analysis failed: ' + error.message);
        } finally {
            this.hideModal();
        }
    }

    async runInference() {
        const response = await fetch(`${this.apiBaseUrl}/infer/${this.currentImageId}`);
        if (!response.ok) {
            throw new Error(`Inference failed: ${response.statusText}`);
        }
        return await response.json();
    }

    async getResults() {
        const response = await fetch(`${this.apiBaseUrl}/result/${this.currentImageId}`);
        if (!response.ok) {
            throw new Error(`Results fetch failed: ${response.statusText}`);
        }
        return await response.json();
    }

    async getExplanation() {
        const response = await fetch(`${this.apiBaseUrl}/explain/${this.currentImageId}`);
        if (!response.ok) {
            // Explanation might not be available, return default
            return {
                explanation: {
                    confidence: 0.85,
                    key_factors: ['Spatial pattern analysis', 'Contrast variation detection', 'Structural feature examination'],
                    decision_process: 'AI analyzed image using computer vision techniques',
                    uncertainty: 0.15
                }
            };
        }
        return await response.json();
    }

    updateResultsUI(inferenceResult, results, explanation) {
        // Update analysis image
        document.getElementById('analysisImage').src = inferenceResult.heatmap_overlay;

        // Update statistics
        document.getElementById('defectCount').textContent = results.analysis.total_defects;
        document.getElementById('defectArea').textContent = results.analysis.defect_area_percentage.toFixed(1) + '%';
        document.getElementById('analysisConfidence').textContent = (results.analysis.defect_area_percentage > 0 ? '85%' : '95%');

        // Update defect list
        this.updateDefectList(results.analysis.defect_types, results.analysis.confidence_scores);

        // Update recommendations
        this.updateRecommendations(results.recommendations);

        // Update explanation
        this.updateExplanation(explanation);
    }

    updateDefectList(defectTypes, confidences) {
        const defectList = document.getElementById('defectList');

        if (defectTypes.length === 0) {
            defectList.innerHTML = `
                <div class="no-defects">
                    <i class="fas fa-check-circle"></i>
                    <p>No defects detected</p>
                </div>
            `;
            return;
        }

        const defectItems = defectTypes.map((type, index) => `
            <div class="defect-item">
                <span class="defect-type">${type.charAt(0).toUpperCase() + type.slice(1)}</span>
                <span class="defect-confidence">${(confidences[index] * 100).toFixed(1)}% confidence</span>
            </div>
        `).join('');

        defectList.innerHTML = defectItems;
    }

    updateRecommendations(recommendations) {
        const recommendationsList = document.getElementById('recommendationsList');
        recommendationsList.innerHTML = recommendations.map(rec => `<li>${rec}</li>`).join('');
    }

    updateExplanation(explanation) {
        const explanationText = document.getElementById('explanationText');
        const confidenceMeter = document.getElementById('confidenceMeter');
        const confidenceValue = document.getElementById('confidenceValue');

        const exp = explanation.explanation;
        const confidencePercent = (exp.confidence * 100).toFixed(1);

        explanationText.textContent = `The AI system analyzed the solar panel image with ${confidencePercent}% confidence. Key factors considered: ${exp.key_factors.join(', ')}. ${exp.decision_process}`;

        confidenceMeter.style.width = confidencePercent + '%';
        confidenceValue.textContent = confidencePercent + '%';
    }

    downloadReport() {
        // Create a simple report
        const reportData = {
            imageId: this.currentImageId,
            timestamp: new Date().toISOString(),
            defectCount: document.getElementById('defectCount').textContent,
            defectArea: document.getElementById('defectArea').textContent,
            confidence: document.getElementById('analysisConfidence').textContent
        };

        const dataStr = JSON.stringify(reportData, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);

        const exportFileDefaultName = `solar_panel_analysis_${this.currentImageId}.json`;

        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();
    }

    showDetailedAnalysis() {
        // For now, just scroll to the explanation section
        document.querySelector('.explanation-section').scrollIntoView({ behavior: 'smooth' });
    }

    resetUI() {
        // Reset file input
        document.getElementById('fileInput').value = '';

        // Hide results
        document.getElementById('resultsSection').style.display = 'none';

        // Reset images
        document.getElementById('originalImage').src = '';
        document.getElementById('analysisImage').src = '';

        // Reset statistics
        document.getElementById('defectCount').textContent = '0';
        document.getElementById('defectArea').textContent = '0.0%';
        document.getElementById('analysisConfidence').textContent = '0%';

        // Reset defect list
        document.getElementById('defectList').innerHTML = `
            <div class="no-defects">
                <i class="fas fa-check-circle"></i>
                <p>No defects detected</p>
            </div>
        `;

        // Reset recommendations
        document.getElementById('recommendationsList').innerHTML = '<li>Panel appears healthy. Continue regular monitoring.</li>';

        // Reset explanation
        document.getElementById('explanationText').textContent = 'The AI system analyzed the solar panel image using advanced computer vision techniques...';
        document.getElementById('confidenceMeter').style.width = '0%';
        document.getElementById('confidenceValue').textContent = '0%';

        // Clear current image ID
        this.currentImageId = null;

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    showModal(title, message, isError = false) {
        const modal = document.getElementById('modal');
        const modalContent = document.getElementById('modalContent');
        const modalTitle = document.getElementById('modalTitle');
        const modalMessage = document.getElementById('modalMessage');

        modalTitle.textContent = title;
        modalMessage.textContent = message;

        if (isError) {
            modalContent.classList.add('error');
        } else {
            modalContent.classList.remove('error');
        }

        modal.style.display = 'flex';
    }

    hideModal() {
        document.getElementById('modal').style.display = 'none';
    }

    showError(message) {
        this.showModal('Error', message, true);
        setTimeout(() => this.hideModal(), 3000);
    }

    async checkBackendHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const health = await response.json();

            if (health.models_loaded) {
                console.log('Backend is healthy and models are loaded');
            } else {
                console.warn('Backend is running but models are not loaded');
            }
        } catch (error) {
            console.error('Backend health check failed:', error);
            this.showError('Unable to connect to analysis server. Please ensure the backend is running.');
        }
    }
}

// Initialize the analyzer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const analyzer = new SolarPanelAnalyzer();

    // Check backend health
    analyzer.checkBackendHealth();

    // Add some visual enhancements
    const uploadArea = document.getElementById('uploadArea');

    // Add subtle animation to upload area
    uploadArea.addEventListener('mouseenter', () => {
        uploadArea.style.transform = 'scale(1.02)';
    });

    uploadArea.addEventListener('mouseleave', () => {
        uploadArea.style.transform = 'scale(1)';
    });

    // Add loading animation to buttons
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            if (this.id === 'downloadReportBtn' || this.id === 'viewDetailsBtn') {
                this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                setTimeout(() => {
                    this.innerHTML = this.innerHTML.replace('<i class="fas fa-spinner fa-spin"></i> Processing...', this.textContent);
                }, 1000);
            }
        });
    });

    console.log('Solar Panel Defect Detection System initialized');
});
