"""
Module 13 - FastAPI Backend (entry point)
============================================
Boots the FastAPI application, wires up CORS for the React frontend,
and mounts the /requirements and /generate routes.

Run with:
    uvicorn app.main:app --reload --port 8000
"""

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure compatibility with langchain_core expecting langchain.debug
import langchain
if not hasattr(langchain, "debug"):
    langchain.debug = False

from app.api.routes import generate
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="Multi-Agent AI Software Workflow Architecture Generation and Risk Prediction Framework",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=Path(settings.diagrams_dir).parent), name="outputs")

app.include_router(generate.router, tags=["pipeline"])


@app.get("/health")
def health_check():
    """Simple liveness probe used by the frontend and deployment tooling."""
    return {"status": "ok", "app": settings.app_name}
