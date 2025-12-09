"""
Data loading utilities for the real-time insider threat pipeline.

This module is deliberately conservative and robust:
- Uses pandas.read_csv with engine="c" when possible (fast, safe)
- Falls back to the Python engine without low_memory when needed
- Validates required columns
- Adds any missing optional columns as "unknown"
- Parses and sorts the timestamp column
"""

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd

from logger import get_logger
from exceptions import PipelineError

logger = get_logger(__name__)


class DataLoader:
    """
    Simple CSV loader with schema validation and safe pandas settings.

    Parameters
    ----------
    required_columns : list of str
        Columns that MUST exist in the CSV (after lower-casing names).
    optional_columns : list of str
        Columns that are nice to have; if missing, they are created with "unknown".
    date_column : str
        Name of the timestamp column (after lower-casing).
    date_format : str or None
        Optional explicit datetime format; if None, pandas auto-detects.
    """

    def __init__(
        self,
        required_columns: List[str],
        optional_columns: Optional[List[str]] = None,
        date_column: str = "date",
        date_format: Optional[str] = None,
    ) -> None:
        self.required_columns = [c.lower() for c in required_columns]
        self.optional_columns = [c.lower() for c in (optional_columns or [])]
        self.date_column = date_column.lower()
        self.date_format = date_format

        logger.info(
            "DataLoader initialized | required=%s | optional=%s | date_column=%s",
            self.required_columns,
            self.optional_columns,
            self.date_column,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load and validate a single CSV file.

        Parameters
        ----------
        file_path : str
            Path to the CSV file.
        nrows : int or None
            If not None, only the first nrows rows are read (used for test_mode).

        Returns
        -------
        pandas.DataFrame
        """
        if not os.path.exists(file_path):
            raise PipelineError(f"Input file does not exist: {file_path}")

        logger.info("Loading CSV: %s (nrows=%s)", file_path, nrows)

        df = self._read_csv_safe(file_path, nrows=nrows)

        # Normalize column names
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Validate columns and add missing optional ones
        df = self._validate_and_fix_columns(df)

        # Parse timestamp column
        df = self._parse_and_sort_dates(df)

        logger.info("Loaded %d rows after preprocessing", len(df))
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _read_csv_safe(self, file_path: str, nrows: Optional[int]) -> pd.DataFrame:
        """
        Read a CSV using pandas with safe options.

        Strategy:
        - Try fast C engine with low_memory=False, on_bad_lines="skip"
        - If that fails (older pandas / weird file), retry with fewer options
        - As a last resort, use engine="python" WITHOUT low_memory
        """
        # First attempt: C engine, low_memory=False
        try:
            logger.info("Reading CSV with engine='c', low_memory=False")
            df = pd.read_csv(
                file_path,
                nrows=nrows,
                engine="c",
                low_memory=False,
                on_bad_lines="skip",  # requires pandas >= 1.3
            )
            return df
        except TypeError:
            # on_bad_lines might not be supported in older pandas
            logger.warning(
                "on_bad_lines not supported in this pandas version; retrying without it"
            )
            try:
                df = pd.read_csv(
                    file_path,
                    nrows=nrows,
                    engine="c",
                    low_memory=False,
                )
                return df
            except Exception as e:
                logger.warning("C engine failed: %s; falling back to python engine", e)

        # Final fallback: Python engine, NO low_memory
        logger.info("Reading CSV with engine='python' (fallback)")
        try:
            df = pd.read_csv(
                file_path,
                nrows=nrows,
                engine="python",
                # DO NOT pass low_memory here – this is what caused the error.
            )
            return df
        except Exception as e:
            logger.error("Failed to read CSV even with python engine: %s", e)
            raise PipelineError(f"Could not read CSV: {e}")

    def _validate_and_fix_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns exist and optional ones are present."""
        missing_required = [
            col for col in self.required_columns if col not in df.columns
        ]
        if missing_required:
            msg = f"Missing required columns: {missing_required}"
            logger.error(msg)
            raise PipelineError(msg)

        # Add missing optional columns
        for col in self.optional_columns:
            if col not in df.columns:
                logger.warning("Optional column '%s' missing; filling with 'unknown'", col)
                df[col] = "unknown"

        return df

    def _parse_and_sort_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the date column and sort by it."""
        if self.date_column not in df.columns:
            msg = f"Date column '{self.date_column}' not found in CSV"
            logger.error(msg)
            raise PipelineError(msg)

        logger.info(
            "Parsing datetime column '%s' (format=%s)",
            self.date_column,
            self.date_format,
        )

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
                infer_datetime_format=True,
            )

        # Drop rows where the date could not be parsed
        before = len(df)
        df = df.dropna(subset=[self.date_column])
        after = len(df)

        if after < before:
            logger.warning(
                "Dropped %d rows with unparsable dates", before - after
            )

        df = df.sort_values(self.date_column).reset_index(drop=True)
        return df
