"""
Global configuration for the Real-Time Insider Threat Detection Project.
"""

# ------------------------------------------------------------
# MODEL SETTINGS
# ------------------------------------------------------------

# TinyLlama model ID from HuggingFace
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Use GPU if available (Streamlit Cloud will still run on CPU)
USE_GPU = True

# Max token length for TinyLlama
MAX_SEQ_LEN = 256
MAX_SEQUENCE_LENGTH = MAX_SEQ_LEN  # alias for compatibility

# ------------------------------------------------------------
# DATA & COLUMN SETTINGS
# ------------------------------------------------------------

# Must-have columns
REQUIRED_COLUMNS = ["date", "user"]

# Optional columns — created automatically if missing
OPTIONAL_COLUMNS = ["pc", "activity", "role"]

# Date parsing
DATE_COLUMN = "date"
DATE_FORMAT = None  # let pandas autodetect

# Load fewer rows when test_mode=True
TEST_MODE_ROWS = 1000

# ------------------------------------------------------------
# ANOMALY DETECTION SETTINGS
# ------------------------------------------------------------

# Ratio of anomalies expected
CONTAMINATION = 0.06
CONTAMINATION_FACTOR = CONTAMINATION  # compatibility alias

# Quantile threshold for anomaly labeling
ANOMALY_QUANTILE = 0.99  # recommended for rolling baseline

# Minimum number of events in a window
MIN_EVENTS_PER_WINDOW = 10

# ------------------------------------------------------------
# ROLLING BASELINE (NEW — required by updated pipeline)
# ------------------------------------------------------------

# Minimum samples required before rolling baseline is used
MIN_BASELINE_SAMPLES = 50

# Maximum number of samples stored in rolling memory
BASELINE_MAX_EVENTS = 5000

# ------------------------------------------------------------
# SLIDING WINDOWS (minutes)
# ------------------------------------------------------------

DEFAULT_WINDOW_MIN = 60
DEFAULT_SLIDE_MIN = 10

# ------------------------------------------------------------
# OUTPUT SETTINGS
# ------------------------------------------------------------

OUTPUT_BASE_DIR = "outputs"
OUTPUT_DIR = OUTPUT_BASE_DIR


