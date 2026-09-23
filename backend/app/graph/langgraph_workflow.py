"""
Module 12 - LangGraph Workflow
=================================
Top-level LangGraph StateGraph that connects the entire pipeline:

    Requirement Agent -> Workflow Agent -> Risk Agent -> Evaluation Agent -> Benchmark Agent

Note: the Benchmark Agent internally re-runs Requirement/Workflow/Risk/
Evaluation once per LLM provider (see Module 10). This top-level graph
represents the SINGLE canonical run used to drive the human-in-the-loop
requirement approval step; once requirements are approved, control
passes to the Benchmark Agent which fans out across all five models
and returns only the best result.
"""

import logging
from typing import Dict, List, TypedDict, Optional

from langgraph.graph import StateGraph, END

from app.agents.requirement_agent import RequirementAgent
from app.agents.workflow_agent import WorkflowAgent
from app.agents.risk_agent import RiskAgent
from app.agents.evaluation_agent import EvaluationAgent
from app.agents.benchmark_agent import BenchmarkAgent
from app.rag.rag_utils import RAGRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    """Shared state threaded through the entire top-level pipeline."""
    project_description: str
    approved_requirements: List[str]
    workflow_json: Dict
    diagram_path: str
    risk_report: List[Dict]
    validation_report: Dict
    best_model: str
    overall_score: float
    comparison_summary: List[Dict]


class MultiAgentPipeline:
    """
    Wires the five agents into one LangGraph graph.

        node: require_requirements  -> (external) human approves requirements
        node: run_workflow          -> Workflow Agent (single reference run)
        node: run_risk              -> Risk Agent (single reference run)
        node: run_evaluation        -> Evaluation Agent (single reference run)
        node: run_benchmark         -> Benchmark Agent (fans out across 5 LLMs,
                                        overwrites workflow/risk/validation with
                                        the winning model's output)
    """

    def __init__(self, retriever: Optional[RAGRetriever] = None):
        self.retriever = retriever
        self.requirement_agent = RequirementAgent(retriever=retriever)
        self.workflow_agent = WorkflowAgent()
        self.risk_agent = RiskAgent(retriever=retriever)
        self.evaluation_agent = EvaluationAgent()
        self.benchmark_agent = BenchmarkAgent(retriever=retriever)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Node: reference workflow generation (used only for a quick preview
    # before the full benchmark fan-out runs)
    # ------------------------------------------------------------------
    def _run_workflow_node(self, state: PipelineState) -> PipelineState:
        """Generate a reference workflow from the approved requirements."""
        result = self.workflow_agent.run(state["approved_requirements"])
        state["workflow_json"] = result["workflow_json"]
        state["diagram_path"] = result["diagram_path"]
        return state

    # ------------------------------------------------------------------
    # Node: reference risk analysis
    # ------------------------------------------------------------------
    def _run_risk_node(self, state: PipelineState) -> PipelineState:
        """Generate a reference risk report from the reference workflow."""
        result = self.risk_agent.run(state["workflow_json"], project_description=state["project_description"])
        state["risk_report"] = result["risk_report"]
        return state

    # ------------------------------------------------------------------
    # Node: reference evaluation
    # ------------------------------------------------------------------
    def _run_evaluation_node(self, state: PipelineState) -> PipelineState:
        """Validate the reference workflow + risk report."""
        state["validation_report"] = self.evaluation_agent.evaluate(state["workflow_json"], state["risk_report"])
        return state

    # ------------------------------------------------------------------
    # Node: benchmark fan-out across all 5 LLMs, select the best
    # ------------------------------------------------------------------
    def _run_benchmark_node(self, state: PipelineState) -> PipelineState:
        """
        Run the full pipeline across all five LLM providers and overwrite
        the pipeline state with ONLY the winning model's output.
        """
        benchmark_result = self.benchmark_agent.run_benchmark(state["project_description"])
        best_output = benchmark_result["best_model_output"]

        state["workflow_json"] = best_output["workflow_json"]
        state["diagram_path"] = best_output["diagram_path"]
        state["risk_report"] = best_output["risk_report"]
        state["validation_report"] = best_output["validation_report"]
        state["best_model"] = benchmark_result["best_model"]
        state["overall_score"] = next(
            item["overall_score"]
            for item in benchmark_result["comparison_summary"]
            if item["model"] == benchmark_result["best_model"]
        )
        state["comparison_summary"] = benchmark_result["comparison_summary"]
        return state

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------
    def _build_graph(self):
        """
        Wire: run_workflow -> run_risk -> run_evaluation -> run_benchmark.
        The Requirement Agent's human-in-the-loop step is handled outside
        this graph (via RequirementAgent.run_initial_generation /
        apply_human_action) since it requires pausing for external input,
        which the FastAPI layer manages across multiple HTTP requests.
        """
        builder = StateGraph(PipelineState)

        builder.add_node("run_workflow", self._run_workflow_node)
        builder.add_node("run_risk", self._run_risk_node)
        builder.add_node("run_evaluation", self._run_evaluation_node)
        builder.add_node("run_benchmark", self._run_benchmark_node)

        builder.set_entry_point("run_workflow")
        builder.add_edge("run_workflow", "run_risk")
        builder.add_edge("run_risk", "run_evaluation")
        builder.add_edge("run_evaluation", "run_benchmark")
        builder.add_edge("run_benchmark", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, project_description: str, approved_requirements: List[str]) -> PipelineState:
        """
        Execute the full pipeline given a project description and a
        human-approved requirements list. Returns the best model's
        workflow, risk report, validation, and benchmark comparison.
        """
        initial_state: PipelineState = {
            "project_description": project_description,
            "approved_requirements": approved_requirements,
            "workflow_json": {},
            "diagram_path": "",
            "risk_report": [],
            "validation_report": {},
            "best_model": "",
            "overall_score": 0.0,
            "comparison_summary": [],
        }
        final_state = self.graph.invoke(initial_state)
        logger.info(
            "Pipeline complete. Best model=%s, overall_score=%.2f",
            final_state["best_model"],
            final_state["overall_score"],
        )
        return final_state
