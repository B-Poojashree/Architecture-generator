"""
Module 4 - FAISS Vector Database
=================================
Builds, persists, and queries FAISS indexes for the GitHub and JIRA
embeddings produced in Module 3.

One VectorStore instance wraps a single FAISS index (either GitHub or
JIRA) so Module 5 (rag_utils.py) can hold two separate VectorStore
objects and query each independently.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import faiss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Thin OOP wrapper around a FAISS IndexFlatIP (inner product, used with
    normalized embeddings to approximate cosine similarity).
    """

    def __init__(self, dataset_tag: str, embedding_dim: int = 384):
        """
        Args:
            dataset_tag: 'github' or 'jira' - used for filenames/logging.
            embedding_dim: Dimension of vectors this index stores.
                384 matches all-MiniLM-L6-v2's output size.
        """
        self.dataset_tag = dataset_tag
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.Index] = None
        self.ids: Optional[np.ndarray] = None
        self.texts: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def create_index(self, embeddings: np.ndarray, ids: np.ndarray, texts: List[str]) -> None:
        """
        Build a new FAISS index from a set of embeddings.

        Args:
            embeddings: (N, embedding_dim) float32 array, ideally
                L2-normalized so inner product == cosine similarity.
            ids: (N,) array of original dataset ids, aligned by row.
            texts: list of N source text strings, aligned by row -
                used to return human-readable context during retrieval.
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}"
            )

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        self.ids = ids
        self.texts = texts

        logger.info(
            "[%s] Built FAISS index with %d vectors (dim=%d)",
            self.dataset_tag,
            self.index.ntotal,
            self.embedding_dim,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_index(self, output_dir: str) -> None:
        """Persist the FAISS index plus its aligned ids/texts metadata."""
        if self.index is None:
            raise RuntimeError("No index to save - call create_index() first")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(out_dir / f"{self.dataset_tag}.index"))
        np.save(out_dir / f"{self.dataset_tag}_ids.npy", self.ids)

        with open(out_dir / f"{self.dataset_tag}_texts.txt", "w", encoding="utf-8") as f:
            for text in self.texts:
                f.write(text.replace("\n", " ") + "\n")

        logger.info("[%s] Saved FAISS index to %s", self.dataset_tag, out_dir)

    def load_index(self, index_dir: str) -> None:
        """Load a previously saved FAISS index plus its metadata."""
        in_dir = Path(index_dir)
        index_path = in_dir / f"{self.dataset_tag}.index"
        ids_path = in_dir / f"{self.dataset_tag}_ids.npy"
        texts_path = in_dir / f"{self.dataset_tag}_texts.txt"

        if not index_path.exists():
            raise FileNotFoundError(f"No FAISS index found at {index_path}")

        self.index = faiss.read_index(str(index_path))
        self.ids = np.load(ids_path, allow_pickle=True)

        with open(texts_path, "r", encoding="utf-8") as f:
            self.texts = [line.rstrip("\n") for line in f.readlines()]

        logger.info("[%s] Loaded FAISS index with %d vectors", self.dataset_tag, self.index.ntotal)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[dict]:
        """
        Search the index for the top_k most similar documents to a
        single query embedding.

        Args:
            query_embedding: (embedding_dim,) or (1, embedding_dim) array.
            top_k: number of nearest neighbours to return.

        Returns:
            List of dicts: [{"id": ..., "text": ..., "score": float}, ...]
            ordered from most to least similar.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded/built - call create_index() or load_index() first")

        query_embedding = np.asarray(query_embedding, dtype="float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                {
                    "id": self.ids[idx],
                    "text": self.texts[idx],
                    "score": float(score),
                }
            )
        return results


def build_index_from_embeddings_dir(embeddings_dir: str, dataset_tag: str, embedding_dim: int = 384) -> VectorStore:
    """
    Convenience function: load the .npy embeddings/ids and .txt texts
    saved by Module 3, then build a VectorStore index from them.
    """
    emb_dir = Path(embeddings_dir)
    embeddings = np.load(emb_dir / f"{dataset_tag}_embeddings.npy")
    ids = np.load(emb_dir / f"{dataset_tag}_ids.npy", allow_pickle=True)

    with open(emb_dir / f"{dataset_tag}_texts.txt", "r", encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f.readlines()]

    store = VectorStore(dataset_tag=dataset_tag, embedding_dim=embedding_dim)
    store.create_index(embeddings, ids, texts)
    return store


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[3]  # backend/
    embeddings_dir = BASE_DIR / "data" / "embeddings"
    vector_db_dir = BASE_DIR / "vector_db"

    for tag, subdir in [("github", "faiss_github_index"), ("jira", "faiss_jira_index")]:
        emb_file = embeddings_dir / f"{tag}_embeddings.npy"
        if emb_file.exists():
            store = build_index_from_embeddings_dir(str(embeddings_dir), dataset_tag=tag)
            store.save_index(str(vector_db_dir / subdir))
        else:
            logger.warning("No embeddings found for '%s' at %s - skipping", tag, emb_file)
