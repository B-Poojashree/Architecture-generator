"""
Module 11 - Evaluation Agent
==============================
Validates the quality of a generated workflow + risk report:
    - missing workflow steps (isolated / unreachable nodes)
    - duplicate tasks
    - incorrect dependencies (cycles, dangling references)
    - workflow completeness (coverage of standard SDLC stages)
    - risk coverage (does every stage have an associated risk entry)

Produces a validation report and a single 0-100 validation score that
the Benchmark Agent uses as one of its scoring dimensions.
"""

import logging
from typing import List, Dict

import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

STANDARD_SDLC_KEYWORDS = ["design", "implement", "develop", "test", "deploy", "review", "plan"]


class EvaluationAgent:
    """
    Stateless validator - takes a workflow_json + risk_report and returns
    a structured validation report. No LangGraph state needed since this
    is a single deterministic pass, but exposed as a class for consistency
    with the other agents and for easy unit testing.
    """

    # ------------------------------------------------------------------
    # Check 1: duplicate tasks
    # ------------------------------------------------------------------
    @staticmethod
    def check_duplicate_tasks(workflow_json: Dict) -> List[str]:
        """Return names of tasks that appear more than once in the workflow."""
        names = [node["name"].strip().lower() for node in workflow_json.get("nodes", [])]
        seen, duplicates = set(), set()
        for name in names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        return list(duplicates)

    # ------------------------------------------------------------------
    # Check 2: incorrect dependencies (cycles + dangling edges)
    # ------------------------------------------------------------------
    @staticmethod
    def check_dependency_integrity(workflow_json: Dict) -> Dict:
        """
        Build a NetworkX graph from the workflow and check for:
            - cycles (invalid for a DAG)
            - edges referencing task ids that don't exist
        """
        dag = nx.DiGraph()
        node_ids = {node["id"] for node in workflow_json.get("nodes", [])}
        dag.add_nodes_from(node_ids)

        dangling_edges = []
        for edge in workflow_json.get("edges", []):
            if edge["from"] not in node_ids or edge["to"] not in node_ids:
                dangling_edges.append(edge)
                continue
            dag.add_edge(edge["from"], edge["to"])

        has_cycle = not nx.is_directed_acyclic_graph(dag)

        return {
            "has_cycle": has_cycle,
            "dangling_edges": dangling_edges,
        }

    # ------------------------------------------------------------------
    # Check 3: missing / unreachable steps
    # ------------------------------------------------------------------
    @staticmethod
    def check_isolated_nodes(workflow_json: Dict) -> List[str]:
        """Find nodes with no incoming or outgoing edges (likely disconnected steps)."""
        node_ids = {node["id"] for node in workflow_json.get("nodes", [])}
        connected = set()
        for edge in workflow_json.get("edges", []):
            connected.add(edge["from"])
            connected.add(edge["to"])
        return list(node_ids - connected)

    # ------------------------------------------------------------------
    # Check 4: SDLC completeness
    # ------------------------------------------------------------------
    @staticmethod
    def check_workflow_completeness(workflow_json: Dict) -> float:
        """
        Estimate what fraction of standard SDLC stages (design, implement,
        test, deploy, etc.) are represented in the workflow's task names.
        Returns a 0.0-1.0 completeness ratio.
        """
        names_text = " ".join(node["name"].lower() for node in workflow_json.get("nodes", []))
        covered = sum(1 for keyword in STANDARD_SDLC_KEYWORDS if keyword in names_text)
        return covered / len(STANDARD_SDLC_KEYWORDS)

    # ------------------------------------------------------------------
    # Check 5: risk coverage
    # ------------------------------------------------------------------
    @staticmethod
    def check_risk_coverage(workflow_json: Dict, risk_report: List[Dict]) -> float:
        """
        Fraction of workflow stages that have at least one corresponding
        entry in the risk report. Returns a 0.0-1.0 coverage ratio.
        """
        stage_names = {node["name"].strip().lower() for node in workflow_json.get("nodes", [])}
        if not stage_names:
            return 0.0
        covered_stages = {entry["stage"].strip().lower() for entry in risk_report}
        return len(stage_names & covered_stages) / len(stage_names)

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------
    def evaluate(self, workflow_json: Dict, risk_report: List[Dict]) -> Dict:
        """
        Run every check and combine results into a single validation
        report + an overall 0-100 validation_score.

        Scoring weights (sum to 100):
            - No cycles / dangling edges:  30 pts
            - No duplicate tasks:          15 pts
            - No isolated nodes:           15 pts
            - Workflow completeness:       20 pts (scaled)
            - Risk coverage:               20 pts (scaled)
        """
        duplicates = self.check_duplicate_tasks(workflow_json)
        dependency_check = self.check_dependency_integrity(workflow_json)
        isolated_nodes = self.check_isolated_nodes(workflow_json)
        completeness = self.check_workflow_completeness(workflow_json)
        risk_coverage = self.check_risk_coverage(workflow_json, risk_report)

        score = 0.0
        score += 30 if not dependency_check["has_cycle"] and not dependency_check["dangling_edges"] else 0
        score += 15 if not duplicates else max(0, 15 - 5 * len(duplicates))
        score += 15 if not isolated_nodes else max(0, 15 - 5 * len(isolated_nodes))
        score += 20 * completeness
        score += 20 * risk_coverage

        report = {
            "duplicate_tasks": duplicates,
            "dependency_issues": dependency_check,
            "isolated_nodes": isolated_nodes,
            "workflow_completeness": round(completeness, 3),
            "risk_coverage": round(risk_coverage, 3),
            "validation_score": round(min(score, 100), 2),
        }

        logger.info("Validation complete - score=%.2f", report["validation_score"])
        return report
