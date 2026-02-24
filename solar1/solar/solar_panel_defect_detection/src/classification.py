import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging
from typing import List, Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DefectClassifier:
    """
    Classifies defect types using KNN on learned embeddings.
    """
    def __init__(self, n_neighbors=5, defect_types=None):
        """
        Initialize the defect classifier.

        Args:
            n_neighbors: Number of neighbors for KNN
            defect_types: List of defect type names
        """
        self.n_neighbors = n_neighbors
        self.defect_types = defect_types or ['crack', 'hotspot', 'cell_damage', 'normal']
        self.knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
        self.is_trained = False

        # Store training embeddings and labels
        self.training_embeddings = []
        self.training_labels = []

        logger.info(f"DefectClassifier initialized with {len(self.defect_types)} classes")

    def extract_embeddings(self, model, images):
        """
        Extract embeddings from images using the trained MoCo model.
        Uses MoCo's extract_features method to get backbone features.

        Args:
            model: Trained MoCo model
            images: List of PIL Images or numpy arrays

        Returns:
            embeddings: numpy array of shape [N, embed_dim]
        """
        model.eval()
        embeddings = []

        with torch.no_grad():
            for img in images:
                # Preprocess image
                if isinstance(img, np.ndarray):
                    # Ensure image is in correct format
                    if img.shape[0] == 3:  # Already CHW format
                        img_tensor = torch.from_numpy(img).float() / 255.0
                    else:  # HWC format
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                else:
                    # Convert PIL to tensor
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    img_tensor = transform(img)

                img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]

                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()

                # Extract features using MoCo's extract_features method
                # This extracts backbone features before the projection head
                if hasattr(model, 'extract_features'):
                    features = model.extract_features(img_tensor)
                elif hasattr(model, 'module') and hasattr(model.module, 'extract_features'):
                    # Handle DataParallel wrapper
                    features = model.module.extract_features(img_tensor)
                else:
                    # Fallback: use encoder_q directly
                    features = model.encoder_q(img_tensor)
                
                embeddings.append(features.cpu().numpy().flatten())

        return np.array(embeddings)


    def train(self, model, training_data):
        """
        Train the KNN classifier on labeled defect examples.

        Args:
            model: Trained feature extraction model
            training_data: Dict with keys 'images' and 'labels'
                          images: List of PIL Images or numpy arrays
                          labels: List of defect type indices
        """
        logger.info("Training defect classifier...")

        images = training_data['images']
        labels = training_data['labels']

        # Extract embeddings
        embeddings = self.extract_embeddings(model, images)

        # Train KNN classifier
        self.knn_classifier.fit(embeddings, labels)
        self.is_trained = True

        # Store for reference
        self.training_embeddings = embeddings
        self.training_labels = labels

        logger.info(f"Trained on {len(images)} samples with {len(np.unique(labels))} classes")

    def predict(self, model, images):
        """
        Predict defect types for new images.

        Args:
            model: Trained feature extraction model
            images: List of PIL Images or numpy arrays

        Returns:
            predictions: List of predicted defect types
            confidences: List of confidence scores
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")

        # Extract embeddings
        embeddings = self.extract_embeddings(model, images)

        # Get predictions and probabilities
        predictions = self.knn_classifier.predict(embeddings)

        # Calculate confidence as distance to nearest neighbor
        distances, indices = self.knn_classifier.kneighbors(embeddings, n_neighbors=self.n_neighbors)
        # Confidence based on inverse of average distance
        confidences = 1 / (1 + distances.mean(axis=1))

        # Convert indices to defect type names
        defect_names = [self.defect_types[pred] for pred in predictions]

        return defect_names, confidences.tolist()

    def evaluate(self, model, test_data):
        """
        Evaluate classifier performance on test data.

        Args:
            model: Trained feature extraction model
            test_data: Dict with keys 'images' and 'labels'

        Returns:
            metrics: Dict with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before evaluation")

        images = test_data['images']
        true_labels = test_data['labels']

        # Get predictions
        pred_labels, confidences = self.predict(model, images)

        # Convert defect names back to indices for evaluation
        pred_indices = [self.defect_types.index(pred) for pred in pred_labels]

        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_indices)
        report = classification_report(true_labels, pred_indices,
                                    target_names=self.defect_types, output_dict=True)

        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': pred_labels,
            'confidences': confidences,
            'true_labels': [self.defect_types[label] for label in true_labels]
        }

        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
        return metrics

class GraphNeuralNetwork(nn.Module):
    """
    Optional Graph Neural Network for defect classification with relational reasoning.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super(GraphNeuralNetwork, self).__init__()
        self.num_layers = num_layers

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(nn.Linear(hidden_dim, hidden_dim))

        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        """
        Forward pass through GNN.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]

        Returns:
            logits: Classification logits [num_nodes, num_classes]
        """
        # Graph convolution (simplified - using mean aggregation)
        for conv in self.convs[:-1]:
            x = conv(x)
            x = F.relu(x)
            x = self.dropout(x)

            # Simple graph aggregation (mean of neighbors)
            # In practice, you'd use proper GNN layers like GCNConv
            if edge_index is not None:
                # Aggregate neighbor features
                row, col = edge_index
                x_agg = torch.zeros_like(x)
                x_agg.index_add_(0, row, x[col])
                degrees = torch.bincount(row, minlength=x.size(0)).float()
                x_agg = x_agg / degrees.unsqueeze(-1).clamp(min=1)
                x = x + x_agg  # Residual connection

        # Final convolution
        x = self.convs[-1](x)
        x = F.relu(x)

        # Classification
        logits = self.classifier(x)
        return logits

class EnsembleDefectClassifier:
    """
    Ensemble classifier combining KNN and GNN approaches.
    """
    def __init__(self, knn_neighbors=5, gnn_hidden_dim=128, defect_types=None):
        self.defect_types = defect_types or ['crack', 'hotspot', 'cell_damage', 'normal']
        self.num_classes = len(self.defect_types)

        # Initialize classifiers
        self.knn_classifier = DefectClassifier(knn_neighbors, self.defect_types)
        self.gnn_classifier = GraphNeuralNetwork(
            input_dim=128,  # Assuming 128-dim embeddings
            hidden_dim=gnn_hidden_dim,
            num_classes=self.num_classes
        )

        self.is_trained = False

    def train(self, model, training_data, use_gnn=True):
        """
        Train ensemble classifier.
        """
        logger.info("Training ensemble defect classifier...")

        # Train KNN classifier
        self.knn_classifier.train(model, training_data)

        if use_gnn:
            # Train GNN classifier (simplified - would need graph construction)
            self._train_gnn(model, training_data)

        self.is_trained = True
        logger.info("Ensemble training completed")

    def _train_gnn(self, model, training_data):
        """
        Train GNN classifier (simplified implementation).
        """
        # Extract embeddings
        embeddings = self.knn_classifier.extract_embeddings(model, training_data['images'])
        labels = training_data['labels']

        # Convert to tensors
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        if torch.cuda.is_available():
            embeddings = embeddings.cuda()
            labels = labels.cuda()
            self.gnn_classifier = self.gnn_classifier.cuda()

        # Simple training (no actual graph structure for this example)
        optimizer = torch.optim.Adam(self.gnn_classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        self.gnn_classifier.train()
        for epoch in range(50):  # Simplified training
            optimizer.zero_grad()

            # Dummy edge index (fully connected graph)
            edge_index = self._create_dummy_edges(embeddings.size(0))

            logits = self.gnn_classifier(embeddings, edge_index)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

        logger.info("GNN training completed")

    def _create_dummy_edges(self, num_nodes):
        """Create dummy fully connected graph edges."""
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])
        return torch.tensor(edges, dtype=torch.long).t()

    def predict(self, model, images, use_ensemble=True):
        """
        Predict using ensemble approach.
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")

        # Get KNN predictions
        knn_preds, knn_confs = self.knn_classifier.predict(model, images)

        if use_ensemble and hasattr(self, 'gnn_classifier'):
            # Get GNN predictions
            embeddings = self.knn_classifier.extract_embeddings(model, images)
            embeddings = torch.tensor(embeddings, dtype=torch.float32)

            if torch.cuda.is_available():
                embeddings = embeddings.cuda()

            self.gnn_classifier.eval()
            with torch.no_grad():
                edge_index = self._create_dummy_edges(embeddings.size(0))
                logits = self.gnn_classifier(embeddings, edge_index)
                gnn_probs = F.softmax(logits, dim=1).cpu().numpy()

            # Ensemble: weighted average
            knn_indices = [self.defect_types.index(pred) for pred in knn_preds]
            ensemble_preds = []

            for i in range(len(images)):
                # Weighted combination
                knn_prob = np.zeros(self.num_classes)
                knn_prob[knn_indices[i]] = knn_confs[i]

                ensemble_prob = 0.7 * knn_prob + 0.3 * gnn_probs[i]
                pred_idx = np.argmax(ensemble_prob)
                ensemble_preds.append(self.defect_types[pred_idx])

            return ensemble_preds, np.max(gnn_probs, axis=1).tolist()
        else:
            return knn_preds, knn_confs

if __name__ == "__main__":
    # Test the classifiers
    print("Testing DefectClassifier...")

    # Create dummy data
    classifier = DefectClassifier()

    # Mock training data
    training_data = {
        'images': [np.random.rand(224, 224, 3) for _ in range(20)],
        'labels': np.random.randint(0, 4, 20).tolist()
    }

    # Mock model (just returns random embeddings)
    class MockModel(nn.Module):
        def forward(self, x):
            return torch.randn(x.size(0), 128)

    mock_model = MockModel()

    # Train classifier
    classifier.train(mock_model, training_data)

    # Test prediction
    test_images = [np.random.rand(224, 224, 3) for _ in range(5)]
    predictions, confidences = classifier.predict(mock_model, test_images)

    print(f"Predictions: {predictions}")
    print(f"Confidences: {confidences}")

    print("DefectClassifier test completed!")
