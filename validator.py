"""
Data Validation Module
Validates data format and content.
"""

import pandas as pd
from typing import List
from logger import get_logger
from exceptions import DataValidationError

logger = get_logger(__name__)


class DataValidator:
    """Validates data format and content."""

    def __init__(
        self,
        required_columns: List[str] = None,
        date_column: str = "date",
    ):
        self.required_columns = required_columns or ["date", "user", "pc", "activity"]
        self.date_column = date_column

    def validate(self, df: pd.DataFrame):
        """Validate DataFrame."""
        if df.empty:
            raise DataValidationError("DataFrame is empty")

        # Check for missing required columns
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")

        lo
