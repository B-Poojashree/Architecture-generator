"""
Module 10 - Benchmark Agent
==============================
Automatically runs the full Requirement -> Workflow -> Risk -> Evaluation
sequence through every registered LLM provider (Gemini, DeepSeek, Llama,
Qwen, Mistral). The user never selects a model.

For each model, computes:
    - Requirement Coverage
    - Workflow Completeness
    - Workflow Correctness
    - Risk Prediction Quality
    - Response Time
    - Overall Workflow Planning Score

Ranks all five, automatically selects the best, and ensures ONLY the
best model's output is returned to the frontend. Full comparison data
is persisted separately for research analysis.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from app.agents.requirement_agent import RequirementAgent
from app.agents.workflow_agent import WorkflowAgent
from app.agents.risk_agent import RiskAgent
from app.agents.evaluation_agent import EvaluationAgent
from app.rag.rag_utils import RAGRetriever
from app.llm import llm_providers

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class BenchmarkAgent:
    """
    Orchestrates a per-model run of the full pipeline and scores each
    run so the single best result can be selected automatically.
    """

    # Weights for the overall workflow planning score (sum to 1.0)
    SCORE_WEIGHTS = {
        "requirement_coverage": 0.25,
        "workflow_completeness": 0.2,
        "workflow_correctness": 0.25,
        "risk_quality": 0.2,
        "response_time": 0.1,  # inverted - faster is better
    }

    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        results_dir: str = "benchmark_results/comparison_logs",
        max_workers: Optional[int] = None,
    ):
        self.retriever = retriever
        self.evaluation_agent = EvaluationAgent()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers or min(4, max(1, len(llm_providers.LLM_PROVIDERS)))

    # ------------------------------------------------------------------
    # Run the full pipeline once, for a single LLM provider
    # ------------------------------------------------------------------
    def _run_pipeline_for_model(self, model_key: str, project_description: str) -> Dict:
        """
        Execute Requirement -> Workflow -> Risk -> Evaluation using a
        single LLM provider, and measure total wall-clock time.
        """
        llm_call = llm_providers.LLM_PROVIDERS[model_key]
        start_time = time.time()

        try:
            requirement_agent = RequirementAgent(retriever=self.retriever, llm_call=llm_call)
            req_state = requirement_agent.run_initial_generation(project_description)
            approved_requirements = req_state["functional_requirements"] + req_state["non_functional_requirements"]

            workflow_agent = WorkflowAgent(llm_call=llm_call)
            workflow_state = workflow_agent.run(approved_requirements)

            risk_agent = RiskAgent(retriever=self.retriever, llm_call=llm_call)
            risk_state = risk_agent.run(workflow_state["workflow_json"], project_description=project_description)

            validation_report = self.evaluation_agent.evaluate(
                workflow_state["workflow_json"], risk_state["risk_report"]
            )

            elapsed = time.time() - start_time

            return {
                "model_key": model_key,
                "success": True,
                "requirements": approved_requirements,
                "workflow_json": workflow_state["workflow_json"],
                "diagram_path": workflow_state["diagram_path"],
                "risk_report": risk_state["risk_report"],
                "validation_report": validation_report,
                "response_time_seconds": round(elapsed, 3),
            }
        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error("[%s] pipeline run failed: %s", model_key, exc)
            return {
                "model_key": model_key,
                "success": False,
                "error": str(exc),
                "response_time_seconds": round(elapsed, 3),
            }

    # ------------------------------------------------------------------
    # Scoring dimensions
    # ------------------------------------------------------------------
    @staticmethod
    def _requirement_coverage(result: Dict) -> float:
        """Score 0-100 based on how many requirements were generated (proxy for coverage)."""
        count = len(result.get("requirements", []))
        return min(100.0, count * 10)  # 10 requirements -> full score

    @staticmethod
    def _workflow_completeness(result: Dict) -> float:
        """Pulled directly from the Evaluation Agent's completeness ratio."""
        return result["validation_report"]["workflow_completeness"] * 100

    @staticmethod
    def _workflow_correctness(result: Dict) -> float:
        """
        Score based on structural correctness: no cycles, no dangling
        edges, no duplicate tasks, no isolated nodes.
        """
        report = result["validation_report"]
        score = 100.0
        if report["dependency_issues"]["has_cycle"]:
            score -= 40
        if report["dependency_issues"]["dangling_edges"]:
            score -= 20
        score -= 5 * len(report["duplicate_tasks"])
        score -= 5 * len(report["isolated_nodes"])
        return max(0.0, score)

    @staticmethod
    def _risk_quality(result: Dict) -> float:
        """Combines risk coverage ratio with presence of mitigation strategies."""
        report = result["validation_report"]
        coverage_score = report["risk_coverage"] * 60
        risk_entries = result.get("risk_report", [])
        has_mitigation = sum(1 for r in risk_entries if r.get("mitigation"))
        mitigation_score = (has_mitigation / len(risk_entries)) * 40 if risk_entries else 0
        return coverage_score + mitigation_score

    @staticmethod
    def _response_time_score(result: Dict, all_times: List[float]) -> float:
        """Faster models score higher, normalized against the slowest run in this batch."""
        if not all_times:
            return 100.0
        slowest = max(all_times) or 1.0
        elapsed = result["response_time_seconds"]
        return max(0.0, 100 * (1 - elapsed / slowest))

    def _compute_overall_score(self, result: Dict, all_times: List[float]) -> Dict:
        """Combine all five sub-scores into a single weighted overall score."""
        scores = {
            "requirement_coverage": self._requirement_coverage(result),
            "workflow_completeness": self._workflow_completeness(result),
            "workflow_correctness": self._workflow_correctness(result),
            "risk_quality": self._risk_quality(result),
            "response_time": self._response_time_score(result, all_times),
        }
        overall = sum(scores[k] * self.SCORE_WEIGHTS[k] for k in scores)
        scores["overall_score"] = round(overall, 2)
        return scores

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run_benchmark(self, project_description: str) -> Dict:
        """
        Run the full pipeline through every registered model, score each
        run, rank them, and return only the best model's output plus a
        redacted comparison summary (scores only, no raw content) for
        the frontend's Model Comparison page.
        """
        all_results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self._run_pipeline_for_model, model_key, project_description): model_key
                for model_key in llm_providers.LLM_PROVIDERS
            }
            for future in as_completed(future_map):
                model_key = future_map[future]
                logger.info("Benchmarking model: %s", model_key)
                try:
                    all_results[model_key] = future.result()
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.exception("Benchmark worker failed for %s: %s", model_key, exc)
                    all_results[model_key] = {
                        "model_key": model_key,
                        "success": False,
                        "error": str(exc),
                        "response_time_seconds": 0.0,
                    }

        successful = {k: v for k, v in all_results.items() if v["success"]}
        if not successful:
            raise RuntimeError("All LLM providers failed during benchmarking")

        all_times = [r["response_time_seconds"] for r in successful.values()]

        scored = {}
        for model_key, result in successful.items():
            scored[model_key] = self._compute_overall_score(result, all_times)

        ranking = sorted(scored.items(), key=lambda item: item[1]["overall_score"], reverse=True)
        best_model_key = ranking[0][0]
        best_result = successful[best_model_key]

        self._persist_comparison(project_description, all_results, scored, best_model_key)

        comparison_summary = [
            {
                "model": model_key,
                **scores,
                "response_time_seconds": all_results[model_key]["response_time_seconds"],
            }
            for model_key, scores in scored.items()
        ]
        comparison_summary.sort(key=lambda x: x["overall_score"], reverse=True)

        logger.info("Best model selected: %s (score=%.2f)", best_model_key, scored[best_model_key]["overall_score"])

        return {
            "best_model": best_model_key,
            "best_model_output": best_result,
            "comparison_summary": comparison_summary,
        }

    # ------------------------------------------------------------------
    # Persistence for research experiments
    # ------------------------------------------------------------------
    def _persist_comparison(
        self, project_description: str, all_results: Dict, scored: Dict, best_model_key: str
    ) -> None:
        """Save the full multi-model comparison to disk for offline research analysis."""
        timestamp = int(time.time())
        record = {
            "timestamp": timestamp,
            "project_description": project_description,
            "results": all_results,
            "scores": scored,
            "best_model": best_model_key,
        }
        out_path = self.results_dir / f"benchmark_{timestamp}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
        logger.info("Persisted benchmark comparison to %s", out_path)
