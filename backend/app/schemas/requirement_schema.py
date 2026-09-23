"""
Pydantic schemas shared across the FastAPI routes.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Body for POST /generate - the initial project description."""
    project_description: str = Field(..., min_length=10, description="Plain-English software project description")


class RequirementActionRequest(BaseModel):
    """Body for POST /requirements/action - a single human-in-the-loop edit."""
    session_id: str
    type: str  # add | delete | modify | approve
    bucket: Optional[str] = "functional"  # functional | non_functional
    text: Optional[str] = None
    index: Optional[int] = None


class RequirementsResponse(BaseModel):
    session_id: str
    functional_requirements: List[str]
    non_functional_requirements: List[str]
    approved: bool


class GenerateResponse(BaseModel):
    """Body for the final POST /generate response after benchmarking."""
    project_description: str
    approved_requirements: List[str]
    workflow_json: Dict
    diagram_path: str
    risk_report: List[Dict]
    validation_report: Dict
    best_model: str
    overall_score: float
    comparison_summary: List[Dict]
