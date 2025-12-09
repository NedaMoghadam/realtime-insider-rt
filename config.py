"""
Global configuration for the Real-Time Insider Threat Detection Project.
"""

# ------------------------------------------------------------
# MODEL SETTINGS
# ------------------------------------------------------------

# TinyLlama model ID from HuggingFace
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Use GPU if available (Streamlit Cloud runs CPU)
USE_GPU = True

# Max token length for TinyLlama
MAX_SEQ_LEN = 256

# Alias required by some parts of the pipeline
MAX_SEQUENCE_LENGTH = MAX_SEQ_LEN

# ------------------------------------------------------------
# DATA & COLUMN SETTINGS
# ------------------------------------------------------------

# 💡 Only "date" and "user" are *required*.
# "pc", "role", "activity" will be handled as optional.
REQUIRED_COLUMNS = ["date", "user"]
OPTIONAL_COLUMNS = ["pc", "role", "activity"]

DATE_COLUMN = "date"
DATE_FORMAT = None  # let pandas infer

# When test_mode=True in the UI, only this many rows are read
TEST_MODE_ROWS = 1000

# ------------------------------------------------------------
# ANOMALY DETECTION
# ------------------------------------------------------------

# Fraction of points IsolationForest will treat as anomalies
# Smaller value → fewer anomalies
CONTAMINATION = 0.02        # 2% expected anomalies

# Quantile threshold for score-based anomaly flag
# Larger value → fewer anomalies
ANOMALY_QUANTILE = 0.98     # top 2% highest scores

# Minimum events in a time window to run detection
MIN_EVENTS_PER_WINDOW = 10

# ------------------------------------------------------------
# SLIDING WINDOWS (minutes)
# ------------------------------------------------------------

DEFAULT_WINDOW_MIN = 60
DEFAULT_SLIDE_MIN = 10

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_BASE_DIR = "outputs"
OUTPUT_DIR = OUTPUT_BASE_DIR  # kept for backward compatibility

# ------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------

SCORE_HISTOGRAM_BINS = 50
PLOT_DPI = 120

