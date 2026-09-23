"""
Module 2 - Dataset Preprocessing
=================================
Loads, cleans, and prepares the two raw datasets used by the RAG layer:
    1. GitHub Repository Dataset
    2. Apache JIRA Issues Dataset

The cleaned outputs are saved to backend/data/processed/ and are the
inputs consumed by Module 3 (embedding_utils.py).

Design notes:
    - Written as a class (DatasetPreprocessor) so it can be reused for
      either dataset by passing a config, instead of duplicating logic.
    - No model training happens here - purely data cleaning.
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class DatasetPreprocessor:
    """
    Generic tabular dataset cleaner.

    Handles the common preprocessing steps needed for both the GitHub
    repository dataset and the Apache JIRA issues dataset:
        - duplicate removal
        - missing value handling
        - date column normalization
        - text column cleaning
        - descriptive statistics
        - persistence of the cleaned dataframe
    """

    def __init__(
        self,
        dataset_name: str,
        text_columns: list,
        date_columns: Optional[list] = None,
    ):
        """
        Args:
            dataset_name: Human readable name, used only for logging.
            text_columns: Columns containing free text (titles, bodies,
                descriptions) that should go through text cleaning.
            date_columns: Columns that should be parsed into datetime.
        """
        self.dataset_name = dataset_name
        self.text_columns = text_columns
        self.date_columns = date_columns or []
        self.df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, path: str) -> pd.DataFrame:
        """Load a CSV/JSON dataset from disk into a pandas DataFrame."""
        path_obj = Path(path)
        logger.info("[%s] Loading dataset from %s", self.dataset_name, path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")

        if path_obj.suffix == ".csv":
            self.df = pd.read_csv(path_obj)
        elif path_obj.suffix == ".json":
            self.df = pd.read_json(path_obj)
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")

        logger.info(
            "[%s] Loaded %d rows, %d columns",
            self.dataset_name,
            *self.df.shape,
        )
        return self.df

    # ------------------------------------------------------------------
    # Cleaning steps
    # ------------------------------------------------------------------
    def remove_duplicates(self) -> "DatasetPreprocessor":
        """Drop exact duplicate rows."""
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        removed = before - len(self.df)
        logger.info("[%s] Removed %d duplicate rows", self.dataset_name, removed)
        return self

    def handle_missing_values(self) -> "DatasetPreprocessor":
        """
        Fill or drop missing values depending on column type.
            - Text columns: fill with empty string.
            - Numeric columns: fill with column median.
            - Rows missing a required identifier are dropped if an
              'id' style column exists.
        """
        for col in self.text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isna().any():
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)

        id_like_cols = [c for c in self.df.columns if c.lower() in ("id", "issue_id", "repo_id")]
        if id_like_cols:
            before = len(self.df)
            self.df = self.df.dropna(subset=id_like_cols).reset_index(drop=True)
            logger.info(
                "[%s] Dropped %d rows missing identifier columns",
                self.dataset_name,
                before - len(self.df),
            )

        logger.info("[%s] Missing value handling complete", self.dataset_name)
        return self

    def convert_date_columns(self) -> "DatasetPreprocessor":
        """Parse configured date columns into pandas datetime objects."""
        for col in self.date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
                logger.info("[%s] Converted date column '%s'", self.dataset_name, col)
        return self

    @staticmethod
    def _clean_text(value: str) -> str:
        """
        Normalize a single text value:
            - lowercase
            - strip URLs
            - strip HTML tags
            - collapse whitespace
            - remove non-alphanumeric noise while keeping basic punctuation
        """
        if not isinstance(value, str):
            return ""
        value = value.lower()
        value = re.sub(r"http\S+|www\.\S+", " ", value)
        value = re.sub(r"<[^>]+>", " ", value)
        value = re.sub(r"[^a-z0-9\s.,!?;:'-]", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def clean_text_columns(self) -> "DatasetPreprocessor":
        """Apply _clean_text to every configured text column."""
        for col in self.text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self._clean_text)
        logger.info("[%s] Text columns cleaned: %s", self.dataset_name, self.text_columns)
        return self

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def display_statistics(self) -> dict:
        """Log and return basic descriptive statistics about the dataset."""
        stats = {
            "dataset_name": self.dataset_name,
            "num_rows": len(self.df),
            "num_columns": len(self.df.columns),
            "columns": list(self.df.columns),
            "null_counts": self.df.isnull().sum().to_dict(),
        }
        logger.info("[%s] Stats: rows=%d, cols=%d", self.dataset_name, stats["num_rows"], stats["num_columns"])
        return stats

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, output_path: str) -> None:
        """Save the cleaned dataframe to disk as CSV."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(out, index=False)
        logger.info("[%s] Saved cleaned dataset to %s", self.dataset_name, out)


def preprocess_github_dataset(raw_path: str, processed_path: str) -> pd.DataFrame:
    """
    Pipeline entry point for the GitHub Repository Dataset.
    Actual columns: name, stars_count, forks_count, watchers, pull_requests,
    primary_language, languages_used, commit_count, created_at, licence
    """
    preprocessor = DatasetPreprocessor(
        dataset_name="GitHub Repository Dataset",
        text_columns=["name", "primary_language", "languages_used", "licence"],
        date_columns=["created_at"],
    )
    preprocessor.load(raw_path)
    preprocessor.remove_duplicates()
    preprocessor.handle_missing_values()
    preprocessor.convert_date_columns()
    preprocessor.clean_text_columns()
    preprocessor.display_statistics()
    preprocessor.save(processed_path)
    return preprocessor.df


def preprocess_jira_dataset(raw_path: str, processed_path: str) -> pd.DataFrame:
    """
    Pipeline entry point for the Apache JIRA Issues Dataset.
    Expected columns (adjust to match your actual raw file):
        issue_id, summary, description, priority, status, created, resolved
    """
    preprocessor = DatasetPreprocessor(
        dataset_name="Apache JIRA Issues Dataset",
        text_columns=["summary", "description", "priority", "status"],
        date_columns=["created", "resolved"],
    )
    preprocessor.load(raw_path)
    preprocessor.remove_duplicates()
    preprocessor.handle_missing_values()
    preprocessor.convert_date_columns()
    preprocessor.clean_text_columns()
    preprocessor.display_statistics()
    preprocessor.save(processed_path)
    return preprocessor.df


def get_default_raw_paths(base_dir: Path) -> tuple[Path, Path]:
    """Return the default raw dataset paths used by the preprocessing pipeline."""
    raw_dir = base_dir / "data" / "raw"
    github_raw = raw_dir / "github_repository_dataset" / "repository_data.csv"
    jira_raw = raw_dir / "apache_jira_issues_dataset" / "issues.csv"
    return github_raw, jira_raw


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]  # backend/
    github_raw, jira_raw = get_default_raw_paths(BASE_DIR)

    github_out = BASE_DIR / "data" / "processed" / "github_cleaned.csv"
    jira_out = BASE_DIR / "data" / "processed" / "jira_cleaned.csv"

    logger.info("Starting dataset preprocessing pipeline")

    if github_raw.exists():
        preprocess_github_dataset(str(github_raw), str(github_out))
    else:
        logger.warning("GitHub raw dataset not found at %s - skipping", github_raw)

    if jira_raw.exists():
        preprocess_jira_dataset(str(jira_raw), str(jira_out))
    else:
        logger.warning("JIRA raw dataset not found at %s - skipping", jira_raw)

    logger.info("Preprocessing pipeline finished")
