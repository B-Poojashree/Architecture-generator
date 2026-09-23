"""Tests for requirement generation expansion and parsing."""

from app.agents.requirement_agent import RequirementAgent


def test_requirement_agent_expands_sparse_output_for_food_project():
    agent = RequirementAgent(retriever=None, llm_call=lambda prompt: type("Resp", (), {"text": '{"functional_requirements": ["Log meals"], "non_functional_requirements": ["Be secure"]}'})())

    state = agent.run_initial_generation("food tracking system")

    combined = state["functional_requirements"] + state["non_functional_requirements"]

    assert len(state["functional_requirements"]) >= 8
    assert len(state["non_functional_requirements"]) >= 5
    assert len(combined) == len({item.lower() for item in combined})
    assert any("food" in item.lower() or "nutrition" in item.lower() or "meal" in item.lower() for item in combined)


def test_requirement_agent_uses_attendance_domain_for_student_system():
    agent = RequirementAgent(retriever=None, llm_call=lambda prompt: type("Resp", (), {"text": '{"functional_requirements": ["Log attendance"], "non_functional_requirements": ["Be secure"]}'})())

    state = agent.run_initial_generation("student attendance management system")

    combined = state["functional_requirements"] + state["non_functional_requirements"]

    assert len(state["functional_requirements"]) >= 8
    assert len(state["non_functional_requirements"]) >= 5
    assert any("attendance" in item.lower() or "student" in item.lower() or "teacher" in item.lower() or "class" in item.lower() for item in combined)
    assert not any("food" in item.lower() or "meal" in item.lower() or "nutrition" in item.lower() for item in combined)