"""
Custom Exceptions for the Real-Time Insider Threat Detection System.
"""

class ThreatDetectionError(Exception):
    """Base exception for all errors in the system."""
    pass


class DataLoadError(ThreatDetectionError):
    """Raised when data loading fails."""
    pass


class DataValidationError(ThreatDetectionError):
    """Raised when validation fails."""
    pass


class ModelLoadError(ThreatDetectionError):
    """Raised when the TinyLlama model fails to load."""
    pass


class InferenceError(ThreatDetectionError):
    """Raised when model inference fails."""
    pass


class AnomalyDetectionError(ThreatDetectionError):
    """Raised when anomaly detection fails."""
    pass


class PipelineError(ThreatDetectionError):
    """Raised when any pipeline step fails."""
    pass
