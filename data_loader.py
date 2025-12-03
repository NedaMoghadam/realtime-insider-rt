"""
Data Loading Module
Handles loading and initial processing of CSV log files.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List

from logger import get_logger
from exceptions import DataLoadError
from validator import DataValidator

logger = get_logger(__name__)


class DataLoader:
    """
    Loads and performs initial processing of log data.
    """

    def __init__(
        self,
        required_columns: List[str] = None,
        optional_columns: List[str] = None,
        date_column: str = "date",
        date_format: Optional[str] = None,
    ):
        self.required_columns = required_columns or ["date", "user", "pc", "activity"]
        self.optional_columns = optional_columns or ["role"]
        self.date_column = date_column
        self.date_format = date_format

        # Attach validator
        self.validator = DataValidator(
            required_columns=self.required_columns,
            date_column=self.date_column,
        )

        logger.info(f"DataLoader initialized with required columns: {self.required_columns}")

    def load(
        self,
        filepath: str,
        nrows: Optional[int] = None,
        encoding: str = "utf-8",
        low_memory: bool = False,
    ) -> pd.DataFrame:
        """Load a CSV file into a DataFrame."""

        filepath = Path(filepath)

        if not filepath.exists():
            raise DataLoadError(f"File not found: {filepath}")

        logger.info(f"Loading data from: {filepath}")

        try:
            df = pd.read_csv(
                filepath,
                nrows=nrows,
                encoding=encoding,
                low_memory=low_memory,
            )

            logger.info(f"Loaded {len(df)} rows.")

            # Validate required columns
            self.validator.validate(df)

            # Clean and enhance
            df = self._initial_processing(df)

            return df

        except Exception as e:
            raise DataLoadError(f"Failed to load CSV: {e}")

    def _initial_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initial cleaning pipeline."""

        # Parse date column
        df = self._parse_dates(df)

        # Add missing optional columns
        for col in self.optional_columns:
            if col not in df.columns:
                df[col] = "unknown"
                logger.info(f"Added missing optional column: {col}")

        # Sort chronologically
        df = df.sort_values(self.date_column).reset_index(drop=True)

        # Remove duplicate rows
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)

        if after < before:
            logger.info(f"Removed {before - after} duplicate rows.")

        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamps safely."""

        try:
            df[self.date_column] = pd.to_datetime(
                df[self.date_column],
                format=self.date_format,
                errors="coerce",
            )

            # Remove invalid dates
            invalid = df[self.date_column].isna().sum()
            if invalid > 0:
                logger.info(f"Removed {invalid} rows with invalid timestamps.")
                df = df.dropna(subset=[self.date_column])

            return df

        except Exception as e:
            raise DataLoadError(f"Date parsing failed: {e}")
