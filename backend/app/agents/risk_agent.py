"""
Module 9 - Risk Analysis Agent
=================================
LangGraph agent that analyzes a generated workflow, retrieves similar
historical issues from the Apache JIRA dataset via RAG, and predicts
risks per workflow stage with severity levels and mitigation strategies.
"""

import json
import logging
from typing import List, Dict, TypedDict, Optional

from langgraph.graph import StateGraph, END

from app.rag.rag_utils import RAGRetriever
from app.llm.llm_providers import call_gemini

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class RiskState(TypedDict):
    """Shared state for the risk analysis sub-graph."""
    project_description: str
    workflow_json: Dict
    jira_context: str
    risk_report: List[Dict]  # [{"stage": "...", "risk": "...", "severity": "High", "mitigation": "..."}]


class RiskAgent:
    """
    Wraps a LangGraph graph with two nodes:
        1. retrieve_jira_context - pulls similar historical issues via RAG
        2. predict_risks          - LLM predicts risks per workflow stage
    """

    SEVERITY_LEVELS = ("High", "Medium", "Low")

    def __init__(self, retriever: Optional[RAGRetriever] = None, llm_call=call_gemini):
        self.retriever = retriever
        self.llm_call = llm_call
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Node: retrieve JIRA context via RAG
    # ------------------------------------------------------------------
    def _retrieve_jira_context_node(self, state: RiskState) -> RiskState:
        """Summarize the workflow and project context, then retrieve similar past issues."""
        workflow_summary = ", ".join(node["name"] for node in state["workflow_json"].get("nodes", []))
        project_description = state.get("project_description", "")
        retrieval_query = f"Project: {project_description}. Workflow: {workflow_summary}".strip()

        if self.retriever is None:
            logger.warning("No retriever configured - proceeding without JIRA context")
            state["jira_context"] = ""
            return state

        context = self.retriever.get_risk_context(retrieval_query)
        state["jira_context"] = context["combined_context"]
        logger.info("Retrieved JIRA context (%d chars) for project-specific risk analysis", len(state["jira_context"]))
        return state

    # ------------------------------------------------------------------
    # Node: predict risks per workflow stage
    # ------------------------------------------------------------------
    def _predict_risks_node(self, state: RiskState) -> RiskState:
        """Prompt the LLM to assign risk severity and mitigation per stage."""
        stage_names = [node["name"] for node in state["workflow_json"].get("nodes", [])]
        stages_text = "\n".join(f"- {s}" for s in stage_names)
        project_description = state.get("project_description", "")

        prompt = f"""You are a software risk analyst specializing in this project.

Project description:
{project_description}

Workflow stages:
{stages_text}

Similar historical issues (for grounding, do not copy verbatim):
{state['jira_context']}

Assess the risks in the context of this exact project. For each workflow stage,
identify the single most likely project-specific risk, not a generic workflow risk.
Focus on real concerns tied to the project description, constraints, users,
security, compliance, integrations, and delivery risks.

Output STRICT JSON, a list of objects with keys:
  "stage": stage name
  "risk": one-sentence risk description
  "severity": one of High, Medium, Low
  "mitigation": one-sentence mitigation strategy

Return ONLY the JSON list, no extra text.
"""
        response = self.llm_call(prompt)
        state["risk_report"] = self._parse_risk_report(response.text)
        logger.info("Generated risk report with %d entries", len(state["risk_report"]))
        return state

    def _parse_risk_report(self, raw_text: str) -> List[Dict]:
        """Parse the LLM's JSON output into a validated list of risk entries."""
        try:
            start = raw_text.find("[")
            end = raw_text.rfind("]") + 1
            report = json.loads(raw_text[start:end])
            for entry in report:
                if entry.get("severity") not in self.SEVERITY_LEVELS:
                    entry["severity"] = "Medium"
            return report
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("Failed to parse risk JSON (%s) - using fallback", exc)
            return [{
                "stage": "Unknown",
                "risk": "Unable to parse risk analysis output",
                "severity": "Medium",
                "mitigation": "Re-run risk analysis with a clearer workflow description",
            }]

    # ------------------------------------------------------------------
    # Helper: surface only the highest-severity stages
    # ------------------------------------------------------------------
    @staticmethod
    def get_high_risk_stages(risk_report: List[Dict]) -> List[Dict]:
        """Filter the risk report down to High severity entries only."""
        return [entry for entry in risk_report if entry.get("severity") == "High"]

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------
    def _build_graph(self):
        """Wire the two nodes into a LangGraph StateGraph."""
        builder = StateGraph(RiskState)
        builder.add_node("retrieve_jira_context", self._retrieve_jira_context_node)
        builder.add_node("predict_risks", self._predict_risks_node)

        builder.set_entry_point("retrieve_jira_context")
        builder.add_edge("retrieve_jira_context", "predict_risks")
        builder.add_edge("predict_risks", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, workflow_json: Dict, project_description: str = "") -> RiskState:
        """Execute the risk analysis graph end to end."""
        initial_state: RiskState = {
            "project_description": project_description,
            "workflow_json": workflow_json,
            "jira_context": "",
            "risk_report": [],
        }
        return self.graph.invoke(initial_state)
