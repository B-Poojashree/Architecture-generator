"""
Module 5 - RAG Module
======================
Given a raw software project description, this module:
    1. Embeds the description with the same Sentence-BERT model used
       to build the FAISS indexes.
    2. Searches the GitHub FAISS index for similar repositories/projects.
    3. Searches the JIRA FAISS index for similar historical issues.
    4. Combines both retrieval results into a single context object
       that downstream LangGraph agents (Requirement, Risk) consume.

This module performs retrieval only - no model training, no fine-tuning.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

from app.rag.embedding_utils import EmbeddingGenerator
from app.rag.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Orchestrates retrieval across both the GitHub and JIRA FAISS indexes
    for a single incoming project description.
    """

    def __init__(
        self,
        github_index_dir: str,
        jira_index_dir: str,
        embedding_model: Optional[EmbeddingGenerator] = None,
    ):
        """
        Args:
            github_index_dir: Directory containing the saved GitHub FAISS index.
            jira_index_dir: Directory containing the saved JIRA FAISS index.
            embedding_model: Optional pre-loaded EmbeddingGenerator to avoid
                reloading Sentence-BERT multiple times across agents.
        """
        self.embedder = embedding_model or EmbeddingGenerator()

        self.github_store = VectorStore(dataset_tag="github", embedding_dim=self.embedder.embedding_dim)
        self.github_store.load_index(github_index_dir)

        self.jira_store = VectorStore(dataset_tag="jira", embedding_dim=self.embedder.embedding_dim)
        self.jira_store.load_index(jira_index_dir)

        logger.info("RAGRetriever initialized with GitHub and JIRA indexes loaded")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def embed_query(self, project_description: str):
        """Convert the raw project description into a query embedding."""
        return self.embedder.encode_texts([project_description], show_progress_bar=False)[0]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve_github_context(self, project_description: str, top_k: int = 5) -> List[Dict]:
        """Retrieve the top_k most similar GitHub repositories/projects."""
        query_embedding = self.embed_query(project_description)
        results = self.github_store.search(query_embedding, top_k=top_k)
        logger.info("Retrieved %d GitHub context records", len(results))
        return results

    def retrieve_jira_context(self, project_description_or_workflow: str, top_k: int = 5) -> List[Dict]:
        """Retrieve the top_k most similar historical JIRA issues."""
        query_embedding = self.embed_query(project_description_or_workflow)
        results = self.jira_store.search(query_embedding, top_k=top_k)
        logger.info("Retrieved %d JIRA context records", len(results))
        return results

    # ------------------------------------------------------------------
    # Combined context for the Requirement Agent
    # ------------------------------------------------------------------
    def get_requirement_context(self, project_description: str, top_k: int = 5) -> Dict:
        """
        Build the context bundle used by the Requirement Agent (Module 7).
        Only GitHub context is relevant at this stage.
        """
        github_matches = self.retrieve_github_context(project_description, top_k=top_k)
        combined_text = self._combine_context(github_matches)
        return {
            "source": "github",
            "matches": github_matches,
            "combined_context": combined_text,
        }

    # ------------------------------------------------------------------
    # Combined context for the Risk Agent
    # ------------------------------------------------------------------
    def get_risk_context(self, workflow_summary: str, top_k: int = 5) -> Dict:
        """
        Build the context bundle used by the Risk Agent (Module 9).
        Only JIRA context is relevant at this stage.
        """
        jira_matches = self.retrieve_jira_context(workflow_summary, top_k=top_k)
        combined_text = self._combine_context(jira_matches)
        return {
            "source": "jira",
            "matches": jira_matches,
            "combined_context": combined_text,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _combine_context(matches: List[Dict]) -> str:
        """
        Flatten a list of retrieved {id, text, score} dicts into a single
        text block suitable for insertion into an LLM prompt.
        """
        lines = []
        for i, match in enumerate(matches, start=1):
            lines.append(f"[{i}] (similarity={match['score']:.3f}) {match['text']}")
        return "\n".join(lines)


def build_default_retriever() -> RAGRetriever:
    """Convenience factory using the standard project directory layout."""
    base_dir = Path(__file__).resolve().parents[2]  # backend/
    return RAGRetriever(
        github_index_dir=str(base_dir / "vector_db" / "faiss_github_index"),
        jira_index_dir=str(base_dir / "vector_db" / "faiss_jira_index"),
    )


if __name__ == "__main__":
    retriever = build_default_retriever()
    sample_description = "A cloud-based inventory management system with real-time stock tracking"

    req_context = retriever.get_requirement_context(sample_description)
    logger.info("Requirement context:\n%s", req_context["combined_context"])

    risk_context = retriever.get_risk_context(sample_description)
    logger.info("Risk context:\n%s", risk_context["combined_context"])
