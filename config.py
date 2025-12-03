"""
Global configuration for the Real-Time Insider Threat Detection Project.
"""

# TinyLlama model ID from HuggingFace
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Use GPU if available
USE_GPU = True

# Max token length for TinyLlama
MAX_SEQ_LEN = 256

# Expected anomaly ratio
CONTAMINATION = 0.06

# Quantile threshold for anomalies
ANOMALY_QUANTILE = 0.84

# Test mode row limit
TEST_MODE_ROWS = 1000

# Sliding windows (minutes)
DEFAULT_WINDOW_MIN = 60
DEFAULT_SLIDE_MIN = 10

# Minimum events needed for processing a window
MIN_EVENTS_PER_WINDOW = 10

# Output folder
OUTPUT_DIR = "outputs"
