"""
Module 8 - Workflow Architecture Agent
========================================
LangGraph agent that converts approved requirements into a complete,
dependency-aware software development workflow, represented as a DAG.

Outputs:
    - workflow nodes (task list)
    - dependency edges
    - workflow JSON (for the API / frontend)
    - architecture diagram (rendered via Graphviz, structure via NetworkX)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, TypedDict, Optional

import networkx as nx
from graphviz import Digraph
from langgraph.graph import StateGraph, END

from app.llm.llm_providers import call_gemini

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """Shared state for the workflow generation sub-graph."""
    approved_requirements: List[str]
    tasks: List[Dict]           # [{"id": "T1", "name": "...", "depends_on": ["T0"]}]
    workflow_json: Dict
    diagram_path: str


class WorkflowAgent:
    """
    Wraps a LangGraph graph with two nodes:
        1. generate_tasks - LLM proposes tasks + dependencies from requirements
        2. build_dag       - validates the DAG with NetworkX and renders it
    """

    def __init__(self, llm_call=call_gemini, output_dir: str = "outputs/architecture_diagrams"):
        self.llm_call = llm_call
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Node: generate task sequence via LLM
    # ------------------------------------------------------------------
    def _generate_tasks_node(self, state: WorkflowState) -> WorkflowState:
        """Ask the LLM to break requirements into dependency-aware tasks."""
        requirements_text = "\n".join(f"- {r}" for r in state["approved_requirements"])
        prompt = f"""You are a software project planner.

Approved requirements:
{requirements_text}

Break this into a software development workflow. Output STRICT JSON,
a list of task objects, each with:
  "id": short string like "T1"
  "name": short task name
  "depends_on": list of task ids this task depends on (empty list if none)

Cover the standard SDLC stages (design, implementation, testing, deployment)
as they apply. Return ONLY the JSON list, no extra text.
"""
        response = self.llm_call(prompt)
        state["tasks"] = self._parse_tasks(response.text)
        logger.info("Generated %d workflow tasks", len(state["tasks"]))
        return state

    @staticmethod
    def _parse_tasks(raw_text: str) -> List[Dict]:
        """Parse the LLM's JSON output into a Python list, with a safe fallback."""
        try:
            start = raw_text.find("[")
            end = raw_text.rfind("]") + 1
            tasks = json.loads(raw_text[start:end])
            for task in tasks:
                task.setdefault("depends_on", [])
            return tasks
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("Failed to parse task JSON (%s) - using fallback single task", exc)
            return [{"id": "T1", "name": "Define project scope", "depends_on": []}]

    # ------------------------------------------------------------------
    # Node: build and validate the DAG, render the diagram
    # ------------------------------------------------------------------
    def _build_dag_node(self, state: WorkflowState) -> WorkflowState:
        """
        Construct a NetworkX DiGraph from the task list, verify it is
        acyclic, then render it to disk with Graphviz.
        """
        dag = nx.DiGraph()
        for task in state["tasks"]:
            dag.add_node(task["id"], name=task["name"])
        for task in state["tasks"]:
            for dep in task["depends_on"]:
                dag.add_edge(dep, task["id"])

        if not nx.is_directed_acyclic_graph(dag):
            logger.warning("Generated workflow contains a cycle - removing back-edges")
            dag = self._break_cycles(dag)

        state["workflow_json"] = {
            "nodes": [{"id": n, "name": dag.nodes[n]["name"]} for n in dag.nodes],
            "edges": [{"from": u, "to": v} for u, v in dag.edges],
        }

        diagram_path = self._render_diagram(dag)
        state["diagram_path"] = diagram_path

        logger.info("Workflow DAG built: %d nodes, %d edges", dag.number_of_nodes(), dag.number_of_edges())
        return state

    @staticmethod
    def _break_cycles(dag: nx.DiGraph) -> nx.DiGraph:
        """Remove edges that create cycles, keeping the graph a valid DAG."""
        while not nx.is_directed_acyclic_graph(dag):
            cycle = next(nx.simple_cycles(dag))
            dag.remove_edge(cycle[-1], cycle[0])
        return dag

    def _render_diagram(self, dag: nx.DiGraph) -> str:
        """Render the DAG to a PNG architecture diagram using Graphviz."""
        dot = Digraph(comment="Workflow Architecture", format="png")
        dot.attr(rankdir="LR")

        for node, attrs in dag.nodes(data=True):
            dot.node(node, attrs.get("name", node))
        for u, v in dag.edges:
            dot.edge(u, v)

        output_path = self.output_dir / "workflow_architecture"
        rendered = dot.render(str(output_path), cleanup=True)
        return str(rendered)

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------
    def _build_graph(self):
        """Wire the two nodes into a LangGraph StateGraph."""
        builder = StateGraph(WorkflowState)
        builder.add_node("generate_tasks", self._generate_tasks_node)
        builder.add_node("build_dag", self._build_dag_node)

        builder.set_entry_point("generate_tasks")
        builder.add_edge("generate_tasks", "build_dag")
        builder.add_edge("build_dag", END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, approved_requirements: List[str]) -> WorkflowState:
        """Execute the workflow generation graph end to end."""
        initial_state: WorkflowState = {
            "approved_requirements": approved_requirements,
            "tasks": [],
            "workflow_json": {},
            "diagram_path": "",
        }
        return self.graph.invoke(initial_state)
