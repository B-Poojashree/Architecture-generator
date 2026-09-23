"""
Module 3 - Embedding Generation
================================
Generates dense vector embeddings for the cleaned GitHub and JIRA
datasets using Sentence-BERT (all-MiniLM-L6-v2), so they can be indexed
by FAISS in Module 4.

Written as a single OOP class (EmbeddingGenerator) so both datasets
reuse the same encoder instance instead of loading the model twice.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingGenerator:
    """
    Wraps a SentenceTransformer model and exposes helper methods to:
        - encode a list of strings into embeddings
        - encode a dataframe column (or combination of columns)
        - persist embeddings + their source ids to disk for FAISS
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None):
        """
        Args:
            model_name: HuggingFace / sentence-transformers model id.
            device: 'cpu' or 'cuda'. If None, sentence-transformers
                auto-detects.
        """
        logger.info("Loading SentenceTransformer model: %s", model_name)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info("Model loaded. Embedding dimension = %d", self.embedding_dim)

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------
    def encode_texts(self, texts: List[str], batch_size: int = 64, show_progress_bar: bool = True) -> np.ndarray:
        """
        Encode a list of raw strings into a (N, embedding_dim) float32
        numpy array, ready for FAISS indexing.
        """
        if not texts:
            raise ValueError("encode_texts received an empty list")

        cleaned = [t if isinstance(t, str) and t.strip() else "empty" for t in texts]

        embeddings = self.model.encode(
            cleaned,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine similarity via inner product
        )
        return embeddings.astype("float32")

    def build_document_text(self, row: pd.Series, columns: List[str]) -> str:
        """
        Concatenate multiple dataframe columns into a single text blob
        that represents one document (e.g. repo name + description, or
        JIRA summary + description).
        """
        parts = [str(row[col]) for col in columns if col in row and pd.notna(row[col])]
        return " . ".join(parts)

    # ------------------------------------------------------------------
    # Dataset-level helpers
    # ------------------------------------------------------------------
    def encode_dataframe(
        self,
        df: pd.DataFrame,
        text_columns: List[str],
        id_column: str,
    ) -> dict:
        """
        Encode every row of a dataframe into embeddings.

        Returns:
            {
                "ids": np.ndarray of source row ids,
                "embeddings": np.ndarray of shape (N, embedding_dim),
                "texts": list of the raw text used per row (for debugging
                         and for returning retrieved context later)
            }
        """
        logger.info("Building document texts from columns: %s", text_columns)
        texts = df.apply(lambda row: self.build_document_text(row, text_columns), axis=1).tolist()

        logger.info("Encoding %d documents", len(texts))
        embeddings = self.encode_texts(texts)

        ids = df[id_column].values if id_column in df.columns else np.arange(len(df))

        return {
            "ids": np.array(ids),
            "embeddings": embeddings,
            "texts": texts,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @staticmethod
    def save_embeddings(result: dict, output_dir: str, dataset_tag: str) -> None:
        """
        Save embeddings, ids, and source texts to disk so Module 4
        (vector_store.py) can load them and build a FAISS index.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / f"{dataset_tag}_embeddings.npy", result["embeddings"])
        np.save(out_dir / f"{dataset_tag}_ids.npy", result["ids"])

        with open(out_dir / f"{dataset_tag}_texts.txt", "w", encoding="utf-8") as f:
            for text in result["texts"]:
                f.write(text.replace("\n", " ") + "\n")

        logger.info(
            "Saved embeddings for '%s' -> %s (%d vectors, dim=%d)",
            dataset_tag,
            out_dir,
            result["embeddings"].shape[0],
            result["embeddings"].shape[1],
        )


def generate_github_embeddings(processed_csv: str, output_dir: str) -> None:
    """Entry point: embed the cleaned GitHub repository dataset."""
    df = pd.read_csv(processed_csv)
    df["repo_id"] = df.index.astype(str)  # no natural id column - use row index

    generator = EmbeddingGenerator()
    result = generator.encode_dataframe(
        df,
        text_columns=["name", "primary_language", "languages_used"],
        id_column="repo_id",
    )
    EmbeddingGenerator.save_embeddings(result, output_dir, dataset_tag="github")


def generate_jira_embeddings(processed_csv: str, output_dir: str) -> None:
    """Entry point: embed the cleaned Apache JIRA issues dataset."""
    df = pd.read_csv(processed_csv)
    generator = EmbeddingGenerator()
    result = generator.encode_dataframe(
        df,
        text_columns=["summary", "description", "priority"],
        id_column="issue_id",
    )
    EmbeddingGenerator.save_embeddings(result, output_dir, dataset_tag="jira")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]  # backend/
    processed_dir = BASE_DIR / "data" / "processed"
    embeddings_dir = BASE_DIR / "data" / "embeddings"

    github_csv = processed_dir / "github_cleaned.csv"
    jira_csv = processed_dir / "jira_cleaned.csv"

    if github_csv.exists():
        generate_github_embeddings(str(github_csv), str(embeddings_dir))
    else:
        logger.warning("Cleaned GitHub dataset not found at %s", github_csv)

    if jira_csv.exists():
        generate_jira_embeddings(str(jira_csv), str(embeddings_dir))
    else:
        logger.warning("Cleaned JIRA dataset not found at %s", jira_csv)
