"""Unit tests for Module 2 - dataset preprocessing."""
from pathlib import Path

import pandas as pd
from app.preprocessing.preprocessing import DatasetPreprocessor, get_default_raw_paths


def test_remove_duplicates():
    df = pd.DataFrame({"id": [1, 1, 2], "text": ["a", "a", "b"]})
    p = DatasetPreprocessor("test", text_columns=["text"])
    p.df = df
    p.remove_duplicates()
    assert len(p.df) == 2


def test_clean_text_lowercases_and_strips_urls():
    cleaned = DatasetPreprocessor._clean_text("Check http://example.com NOW")
    assert "http" not in cleaned
    assert cleaned == cleaned.lower()


def test_default_raw_paths_use_existing_dataset_files():
    base_dir = Path(__file__).resolve().parents[1]
    github_raw, jira_raw = get_default_raw_paths(base_dir)

    assert github_raw.name == "repository_data.csv"
    assert jira_raw.name == "issues.csv"
    assert github_raw.exists()
    assert jira_raw.exists()
