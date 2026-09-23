"""
Module 13 - FastAPI Backend (routes)
=======================================
POST /requirements/start   - kicks off requirement generation for a project description
POST /requirements/action  - applies a human-in-the-loop edit (add/delete/modify/approve)
POST /generate              - runs the full pipeline once requirements are approved,
                               returns requirements, workflow, diagram, risk analysis,
                               validation, best model, and overall score
"""

import uuid
import logging
from fastapi import APIRouter, HTTPException

from app.schemas.requirement_schema import (
    GenerateRequest,
    RequirementActionRequest,
    RequirementsResponse,
    GenerateResponse,
)
from app.agents.requirement_agent import RequirementAgent
from app.graph.langgraph_workflow import MultiAgentPipeline
from app.rag.rag_utils import build_default_retriever
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory session store: session_id -> {"state": RequirementState, "description": str}
# NOTE: for production, replace with Redis or a database-backed session store.
_SESSIONS = {}

try:
    _retriever = build_default_retriever()
except Exception as exc:
    logger.warning("RAG retriever unavailable at startup (%s) - agents will run without RAG context", exc)
    _retriever = None

_requirement_agent = RequirementAgent(retriever=_retriever)
_pipeline = MultiAgentPipeline(retriever=_retriever)


@router.post("/requirements/start", response_model=RequirementsResponse)
def start_requirements(payload: GenerateRequest):
    """Generate an initial requirement draft for human review."""
    state = _requirement_agent.run_initial_generation(payload.project_description)
    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = {"state": state, "description": payload.project_description}

    return RequirementsResponse(
        session_id=session_id,
        functional_requirements=state["functional_requirements"],
        non_functional_requirements=state["non_functional_requirements"],
        approved=state["approved"],
    )


@router.post("/requirements/action", response_model=RequirementsResponse)
def apply_requirement_action(payload: RequirementActionRequest):
    """Apply a single add/delete/modify/approve action to a requirement session."""
    session = _SESSIONS.get(payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    action = {"type": payload.type, "bucket": payload.bucket, "text": payload.text, "index": payload.index}
    session["state"] = _requirement_agent.apply_human_action(session["state"], action)

    return RequirementsResponse(
        session_id=payload.session_id,
        functional_requirements=session["state"]["functional_requirements"],
        non_functional_requirements=session["state"]["non_functional_requirements"],
        approved=session["state"]["approved"],
    )


@router.post("/generate", response_model=GenerateResponse)
def generate(session_id: str):
    """
    Run the full LangGraph pipeline (Workflow -> Risk -> Evaluation -> Benchmark)
    for an already-approved requirement session, and return the winning model's output.
    """
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session["state"]["approved"]:
        raise HTTPException(status_code=400, detail="Requirements have not been approved yet")

    approved_requirements = (
        session["state"]["functional_requirements"] + session["state"]["non_functional_requirements"]
    )

    result = _pipeline.run(session["description"], approved_requirements)

    return GenerateResponse(
        project_description=session["description"],
        approved_requirements=approved_requirements,
        workflow_json=result["workflow_json"],
        diagram_path=result["diagram_path"],
        risk_report=result["risk_report"],
        validation_report=result["validation_report"],
        best_model=result["best_model"],
        overall_score=result["overall_score"],
        comparison_summary=result["comparison_summary"],
    )
