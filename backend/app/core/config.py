"""
Core application configuration.
Centralizes paths and settings so agents/routes don't hardcode strings.
"""

from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parents[2]  # backend/


class Settings(BaseSettings):
    """Application-wide settings, overridable via environment variables."""

    app_name: str = "Multi-Agent Workflow Risk Framework API"
    github_index_dir: str = str(BASE_DIR / "vector_db" / "faiss_github_index")
    jira_index_dir: str = str(BASE_DIR / "vector_db" / "faiss_jira_index")
    diagrams_dir: str = str(BASE_DIR / "outputs" / "architecture_diagrams")
    workflow_json_dir: str = str(BASE_DIR / "outputs" / "workflow_json")
    benchmark_results_dir: str = str(BASE_DIR / "benchmark_results" / "comparison_logs")
    cors_origins: list = ["http://localhost:3000", "http://localhost:3001"]

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
