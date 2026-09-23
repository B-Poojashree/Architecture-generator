"""Unit tests for Module 11 - evaluation agent scoring logic."""
from app.agents.evaluation_agent import EvaluationAgent
from app.agents.risk_agent import RiskAgent

SAMPLE_WORKFLOW = {
    "nodes": [{"id": "T1", "name": "Design system"}, {"id": "T2", "name": "Implement API"}],
    "edges": [{"from": "T1", "to": "T2"}],
}
SAMPLE_RISK_REPORT = [
    {"stage": "Design system", "risk": "Scope creep", "severity": "Medium", "mitigation": "Freeze requirements early"},
]


def test_no_duplicates_no_cycles():
    agent = EvaluationAgent()
    report = agent.evaluate(SAMPLE_WORKFLOW, SAMPLE_RISK_REPORT)
    assert report["duplicate_tasks"] == []
    assert report["dependency_issues"]["has_cycle"] is False
    assert report["validation_score"] > 0


def test_risk_prompt_uses_project_description():
    captured = {}

    def fake_llm_call(prompt):
        captured["prompt"] = prompt
        class Response:
            text = "[{\"stage\": \"Design system\", \"risk\": \"Scope creep\", \"severity\": \"Medium\", \"mitigation\": \"Freeze requirements early\"}]"
        return Response()

    agent = RiskAgent(llm_call=fake_llm_call, retriever=None)
    project_description = "Build a secure payment API for a fintech app with PCI compliance and OAuth login."

    agent.run(SAMPLE_WORKFLOW, project_description=project_description)

    prompt = captured["prompt"]
    assert "secure payment API" in prompt.lower()
    assert "pci compliance" in prompt.lower()
    assert "oauth login" in prompt.lower()
