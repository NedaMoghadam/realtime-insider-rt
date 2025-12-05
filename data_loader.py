"""
Data Loading Module
Handles loading and initial processing of CSV log files.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

from logger import get_logger
from exceptions import DataLoadError
from validator import DataValidator
import config

logger = get_logger(__name__)


class DataLoader:
    """
    Loads and performs initial processing of log data.
    """

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        optional_columns: Optional[List[str]] = None,
        date_column: str = "date",
        date_format: Optional[str] = None,
    ):
        """
        Initialize data loader.
        """
        # Use config defaults if not provided
        self.required_columns = required_columns or config.REQUIRED_COLUMNS
        self.optional_columns = optional_columns or config.OPTIONAL_COLUMNS
        self.date_column = date_column
        self.date_format = date_format

        self.validator = DataValidator(
            required_columns=self.required_columns,
            date_column=date_column,
        )

        logger.info(
            f"DataLoader initialized with required columns: {self.required_columns}"
        )

    def load(
        self,
        filepath: str,
        nrows: Optional[int] = None,
        encoding: str = "utf-8",
        low_memory: bool = False,
    ) -> pd.DataFrame:
        """
        Load data from CSV file.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise DataLoadError(f"File not found: {filepath}")

        logger.info(f"Loading data from {filepath}")
        if nrows:
            logger.info(f"Loading first {nrows} rows only")

        try:
            df = pd.read_csv(
                filepath,
                nrows=nrows,
                encoding=encoding,
                low_memory=low_memory,
            )

            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

            # Validate
            self.validator.validate(df)

            # Process
            df = self._initial_processing(df)

            logger.info(f"Data loading complete: {len(df)} valid rows")

            return df

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise DataLoadError(f"Could not load {filepath}: {e}")

    def _initial_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform initial processing on loaded data.
        """
        logger.debug("Performing initial processing...")

        # Convert date column
        df = self._parse_dates(df)

        # ---------- NEW: handle aggregated daily data ----------
        # If 'pc' or 'activity' are missing, create them from other columns
        if ("pc" not in df.columns) or ("activity" not in df.columns):
            logger.info(
                "No 'pc'/'activity' columns found. "
                "Assuming aggregated daily user data and auto-building them."
            )

            # Synthetic PC if not present
            if "pc" not in df.columns:
                df["pc"] = "AGG_PC"

            # Build a daily activity summary if missing
            if "activity" not in df.columns:
                def build_activity(row):
                    # ignore these when building text
                    ignore = {self.date_column, "user", "pc", "role"}
                    parts = []
                    for col in row.index:
                        if col in ignore:
                            continue
                        value = row[col]
                        if pd.isna(value):
                            continue
                        parts.append(f"{col}={value}")
                    # Example: "emails=5 | downloads=2 | risky_logons=1"
                    return " | ".join(parts) if parts else "DailySummary"

                df["activity"] = df.apply(build_activity, axis=1)
        # ---------- end NEW block -------------------------------

        # Add optional columns if missing
        for col in self.optional_columns:
            if col not in df.columns:
                df[col] = "unknown"
                logger.debug(f"Added missing optional column: {col}")

        # Sort by date
        df = df.sort_values(self.date_column).reset_index(drop=True)

        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates()
        if len(df) < original_len:
            logger.warning(f"Removed {original_len - len(df)} duplicate rows")

        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse date column.
        """
        logger.debug(f"Parsing date column: {self.date_column}")

        try:
            if self.date_format:
                df[self.date_column] = pd.to_datetime(
                    df[self.date_column],
                    format=self.date_format,
                    errors="coerce",
                )
            else:
                df[self.date_column] = pd.to_datetime(
                    df[self.date_column],
                    errors="coerce",
                )

            invalid_count = df[self.date_column].isna().sum()
            if invalid_count > 0:
                logger.warning(
                    f"Found {invalid_count} invalid dates, dropping those rows"
                )
                df = df.dropna(subset=[self.date_column])

            logger.debug("Date parsing complete")
            return df

        except Exception as e:
            logger.error(f"Date parsing failed: {e}")
            raise DataLoadError(f"Could not parse dates: {e}")

