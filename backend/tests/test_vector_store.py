"""Unit tests for Module 4 - FAISS vector store."""
import numpy as np
from app.rag.vector_store import VectorStore


def test_create_and_search_index():
    store = VectorStore(dataset_tag="test", embedding_dim=4)
    embeddings = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype="float32")
    ids = np.array([10, 20, 30])
    texts = ["doc a", "doc b", "doc c"]
    store.create_index(embeddings, ids, texts)

    results = store.search(np.array([1, 0, 0, 0], dtype="float32"), top_k=1)
    assert results[0]["id"] == 10
    assert results[0]["text"] == "doc a"
