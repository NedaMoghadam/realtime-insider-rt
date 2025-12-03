"""
Anomaly Detection Module
Implements Isolation Forest-based anomaly detection on embeddings.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict
import warnings

from logger import get_logger
from exceptions import AnomalyDetectionError

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class AnomalyDetector:
    """
    Anomaly detector using Isolation Forest.
    """

    def __init__(
        self,
        contamination: float = 0.06,
        random_state: int = 42,
        n_estimators: int = 100,
        max_samples: str = "auto",
        normalize: bool = True,
    ):
        """
        Initialize anomaly detector.

        Args:
            contamination: Expected proportion of anomalies
            random_state: Random seed for reproducibility
            n_estimators: Number of trees in the forest
            max_samples: Number of samples to draw for each tree
            normalize: Whether to normalize features
        """
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.normalize = normalize

        # Core models
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples=max_samples,
        )

        self.scaler = StandardScaler() if normalize else None
        self.is_fitted = False

        logger.info(f"AnomalyDetector initialized with contamination={contamination}")

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        """
        Fit the anomaly detector on data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
        """
        if X.shape[0] == 0:
            raise AnomalyDetectionError("Cannot fit on empty data")

        if X.shape[0] < 2:
            raise AnomalyDetectionError("Need at least 2 samples to fit")

        try:
            logger.debug(f"Fitting on {X.shape[0]} samples with {X.shape[1]} features")

            # Normalize if needed
            if self.normalize:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X

            self.iso_forest.fit(X_scaled)
            self.is_fitted = True

            logger.debug("Anomaly detector fitted successfully")
            return self

        except Exception as e:
            logger.error(f"Failed to fit anomaly detector: {e}")
            raise AnomalyDetectionError(f"Fitting failed: {e}")

    def predict(
        self,
        X: np.ndarray,
        return_scores: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies in data.

        Args:
            X: Feature matrix (n_samples, n_features)
            return_scores: Whether to return scores

        Returns:
            scores: higher = more anomalous
            labels: 1 = anomaly, 0 = normal
        """
        if not self.is_fitted:
            raise AnomalyDetectionError("Detector must be fitted before prediction")

        if X.shape[0] == 0:
            return np.array([]), np.array([])

        try:
            if self.normalize:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X

            # IsolationForest decision_function: higher = more normal
            scores = -self.iso_forest.decision_function(X_scaled)
            labels = self.iso_forest.predict(X_scaled)
            labels = (labels == -1).astype(int)  # -1 → anomaly → 1

            logger.debug(f"Predicted {labels.sum()} anomalies out of {len(labels)} samples")

            if return_scores:
                return scores, labels
            else:
                return labels

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise AnomalyDetectionError(f"Prediction failed: {e}")

    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and predict in one step.

        Args:
            X: Feature matrix

        Returns:
            scores, labels
        """
        self.fit(X)
        return self.predict(X)

    def get_threshold(self, scores: np.ndarray, quantile: float = 0.84) -> float:
        """
        Get threshold from anomaly scores.

        Args:
            scores: Anomaly scores
            quantile: Quantile (0–1)

        Returns:
            threshold value
        """
        if len(scores) == 0:
            return 0.0

        thr = np.quantile(scores, quantile)
        logger.debug(f"Computed threshold {thr:.4f} at {quantile*100:.0f}th percentile")
        return float(thr)


class NodeAnomalyDetector:
    """
    Detector for node-level anomalies.
    Aggregates edge-level features to node level.
    """

    def __init__(
        self,
        contamination: float = 0.06,
        aggregation: str = "mean",
    ):
        """
        Args:
            contamination: expected proportion of anomalous nodes
            aggregation: "mean", "max", or "sum"
        """
        self.contamination = contamination
        self.aggregation = aggregation
        self.detector = AnomalyDetector(contamination=contamination)

        logger.info(f"NodeAnomalyDetector initialized with aggregation={aggregation}")

    def aggregate_to_nodes(
        self,
        edge_features: np.ndarray,
        node_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate edge features to node-level features.

        Returns:
            node_features, unique_node_ids
        """
        unique_nodes = np.unique(node_ids)
        node_features = []

        for node in unique_nodes:
            mask = node_ids == node
            node_edges = edge_features[mask]

            if self.aggregation == "mean":
                feat = node_edges.mean(axis=0)
            elif self.aggregation == "max":
                feat = node_edges.max(axis=0)
            elif self.aggregation == "sum":
                feat = node_edges.sum(axis=0)
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation}")

            node_features.append(feat)

        node_features = np.array(node_features)
        logger.debug(f"Aggregated {len(edge_features)} edges into {len(unique_nodes)} nodes")

        return node_features, unique_nodes

    def fit_predict_nodes(
        self,
        edge_features: np.ndarray,
        node_ids: np.ndarray,
    ) -> Dict:
        """
        Detect anomalous nodes.

        Returns:
            dict with node_ids, scores, labels, n_anomalies
        """
        node_features, uniq_nodes = self.aggregate_to_nodes(edge_features, node_ids)
        scores, labels = self.detector.fit_predict(node_features)

        result = {
            "node_ids": uniq_nodes,
            "scores": scores,
            "labels": labels,
            "n_anomalies": int(labels.sum()),
        }

        logger.info(
            f"Detected {result['n_anomalies']} anomalous nodes out of {len(uniq_nodes)}"
        )

        return result


if __name__ == "__main__":
    # Simple self-test
    print("Testing AnomalyDetector with synthetic data...")

    np.random.seed(42)
    n_samples = 500
    n_features = 16

    normal = np.random.randn(int(n_samples * 0.94), n_features)
    anomalies = np.random.randn(int(n_samples * 0.06), n_features) * 3 + 5
    X = np.vstack([normal, anomalies])

    detector = AnomalyDetector(contamination=0.06)
    scores, labels = detector.fit_predict(X)

    print(f"Total samples: {len(X)}")
    print(f"Predicted anomalies: {labels.sum()}")
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
