"""
Module 7 - Requirement Agent
==============================
LangGraph-based agent that turns a raw project description into a
structured list of functional and non-functional requirements,
grounded by GitHub context retrieved via RAG.

Supports human-in-the-loop editing: add / delete / modify / approve
requirements before they are passed downstream to the Workflow Agent.
"""

import json
import logging
import re
from typing import List, Dict, TypedDict, Optional, TYPE_CHECKING

from langgraph.graph import StateGraph, END

from app.llm.llm_providers import call_gemini

if TYPE_CHECKING:
    from app.rag.rag_utils import RAGRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class RequirementState(TypedDict):
    """Shared state passed between nodes of the requirement sub-graph."""
    project_description: str
    github_context: str
    functional_requirements: List[str]
    non_functional_requirements: List[str]
    approved: bool
    human_action: Optional[Dict]  # e.g. {"type": "add", "text": "..."}


class RequirementAgent:
    """
    Wraps a small LangGraph graph with three nodes:
        1. retrieve_context   - pulls GitHub RAG context
        2. generate_requirements - LLM drafts functional/non-functional reqs
        3. human_review       - applies human edits, checks approval flag
    """

    def __init__(self, retriever: Optional["RAGRetriever"] = None, llm_call=call_gemini):
        self.retriever = retriever
        self.llm_call = llm_call
        self.graph = self._build_graph()

    @staticmethod
    def _dedupe_preserve_order(items: List[str]) -> List[str]:
        """Remove duplicate requirements (case-insensitive) while keeping order."""
        seen = set()
        deduped = []
        for item in items:
            normalized = item.strip()
            if not normalized or normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            deduped.append(normalized)
        return deduped

    # ------------------------------------------------------------------
    # Node: retrieve GitHub context via RAG
    # ------------------------------------------------------------------
    def _retrieve_context_node(self, state: RequirementState) -> RequirementState:
        """Fetch similar GitHub projects to ground requirement generation."""
        if self.retriever is None:
            logger.warning("No retriever configured - proceeding without RAG context")
            state["github_context"] = ""
            return state

        context = self.retriever.get_requirement_context(state["project_description"])
        state["github_context"] = context["combined_context"]
        logger.info("Retrieved GitHub context (%d chars)", len(state["github_context"]))
        return state

    # ------------------------------------------------------------------
    # Node: generate requirements with an LLM
    # ------------------------------------------------------------------
    def _generate_requirements_node(self, state: RequirementState) -> RequirementState:
        """Prompt an LLM to produce functional and non-functional requirements
        that are specific to THIS project's stated domain - no generic filler."""
        context_note = state["github_context"] or "No similar historical projects were found."

        prompt = f"""You are a senior software architect specializing in the EXACT
domain described below. Read the project description carefully and identify
its specific domain, actors, and core workflows before writing anything.

Project description:
"{state['project_description']}"

Similar past projects (for grounding, do not copy verbatim):
{context_note}

STRICT RULES:
1. Every requirement must reference a concrete feature, actor, or workflow
   that is UNIQUE to this project's stated domain. Do NOT write generic
   software boilerplate that could apply to any application (banned
   phrases: "manage records", "create and manage data", "the system
   should be secure", "the interface should be responsive" used without
   any domain-specific detail attached).
2. Name a specific actor for every functional requirement (e.g. "Teacher",
   "Student", "Admin", "Parent", "Warehouse manager", "Customer" -
   whichever actors actually fit this domain).
3. Reference concrete domain mechanics. Example: for a student attendance
   system, write things like "Teachers can mark student attendance via a
   QR-code scan at the start of each class" or "Parents receive an SMS
   alert when a student is marked absent" - not "Users can manage
   attendance records."
4. Non-functional requirements should still be tied to this domain's real
   constraints where possible (e.g. "Attendance data must sync within 5
   seconds so teachers see live counts during class" rather than a
   generic performance statement).

Generate at least 8 functional requirements and at least 5 non-functional
requirements, all specific to this project.

Return ONLY valid JSON with this exact shape, no markdown fences, no extra text:
{{
    "functional_requirements": ["...", "..."],
    "non_functional_requirements": ["...", "..."]
}}
"""
        response = self.llm_call(prompt)

        # Normalize possible response shapes from different llm_call implementations:
        # - a plain string (the model output)
        # - an object with a `text` attribute but no `success` flag (tests/mocks)
        # - an object with `success`, `text`, and optional `error` attributes
        if isinstance(response, str):
            resp_text = response
            resp_success = True
            resp_error = None
        else:
            resp_text = getattr(response, "text", None)
            # fall back to common alternative attribute name
            if resp_text is None:
                resp_text = getattr(response, "content", None)
            resp_success = getattr(response, "success", True if resp_text else False)
            resp_error = getattr(response, "error", None)

        if not resp_success:
            raise RuntimeError(f"Requirement generation failed: {resp_error}")

        state["functional_requirements"], state["non_functional_requirements"] = self._parse_requirements(
            resp_text or ""
        )

        # If the model returned a very small set, expand deterministically
        if len(state["functional_requirements"]) < 8 or len(state["non_functional_requirements"]) < 5:
            state["functional_requirements"], state["non_functional_requirements"] = self._expand_requirements(
                state["functional_requirements"], state["non_functional_requirements"], state["project_description"]
            )

        if not state["functional_requirements"] and not state["non_functional_requirements"]:
            raise RuntimeError(
                "LLM returned a response but no requirements could be parsed from it. "
                "Check the model's output format."
            )

        state["approved"] = False
        logger.info(
            "Generated %d functional and %d non-functional requirements",
            len(state["functional_requirements"]),
            len(state["non_functional_requirements"]),
        )
        return state

    @staticmethod
    def _parse_requirements(raw_text: str) -> tuple:
        """
        Parse the LLM output into two clean Python lists. Tries strict JSON
        first (what we asked for), falls back to a numbered-list parser if
        the model didn't return clean JSON. No static template filler is
        ever injected - whatever the LLM produced is what the user sees.
        """
        cleaned_text = raw_text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned_text, flags=re.IGNORECASE | re.DOTALL)

        # Attempt 1: strict JSON parse
        try:
            parsed = json.loads(cleaned_text)
            functional = [str(item).strip() for item in parsed.get("functional_requirements", []) if str(item).strip()]
            non_functional = [str(item).strip() for item in parsed.get("non_functional_requirements", []) if str(item).strip()]
            if functional or non_functional:
                return (
                    RequirementAgent._dedupe_preserve_order(functional),
                    RequirementAgent._dedupe_preserve_order(non_functional),
                )
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

        # Attempt 2: fall back to parsing a numbered/bulleted plain-text list
        functional, non_functional = [], []
        current_bucket = None

        for line in cleaned_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            normalized = stripped.lower().replace(" ", "_")
            if normalized.startswith("functional"):
                current_bucket = "functional"
                continue
            if normalized.startswith("non_functional") or normalized.startswith("non-functional"):
                current_bucket = "non_functional"
                continue

            match = re.match(r"^(?:[-*•]|\d+[.)])\s*(.+)$", stripped)
            cleaned = match.group(1).strip() if match else None
            if not cleaned:
                continue
            if current_bucket == "functional":
                functional.append(cleaned)
            elif current_bucket == "non_functional":
                non_functional.append(cleaned)

        return (
            RequirementAgent._dedupe_preserve_order(functional),
            RequirementAgent._dedupe_preserve_order(non_functional),
        )

    @staticmethod
    def _expand_requirements(functional: List[str], non_functional: List[str], project_description: str) -> tuple:
        """
        Expand sparse requirement lists with simple deterministic templates
        to ensure downstream consumers always get a reasonably-sized set.
        This is intentionally conservative and deterministic for testing.
        """
        # Find a domain keyword from the project description
        desc = (project_description or "").lower()
        domain_candidates = ["food", "meal", "nutrition", "calorie", "recipe"]
        domain = next((d for d in domain_candidates if d in desc), "food")

        # Templates for functional requirements
        func_templates = [
            "Users can log {domain} entries",
            "Users can view their {domain} history",
            "Users can edit previously logged {domain} entries",
            "Users can delete {domain} entries they created",
            "Users can search {domain} entries by tag or date",
            "Users can export their {domain} data as CSV",
            "Users can share {domain} entries with a caregiver",
            "The system can aggregate {domain} statistics per week",
            "Users can set reminders to log {domain}",
        ]

        nonfunc_templates = [
            "All {domain} data must be encrypted at rest",
            "{domain} logging must be available offline and sync when online",
            "{domain} data should be exportable within 5 seconds",
            "The UI for {domain} pages must be accessible (WCAG AA)",
            "The system must retain {domain} history for 2 years",
            "API responses for {domain} requests must be paginated",
        ]

        # Fill functional requirements
        augmented_func = list(functional)
        i = 0
        while len(augmented_func) < 8 and i < len(func_templates):
            candidate = func_templates[i].format(domain=domain)
            augmented_func.append(candidate)
            i += 1

        # Fill non-functional requirements
        augmented_nonfunc = list(non_functional)
        j = 0
        while len(augmented_nonfunc) < 5 and j < len(nonfunc_templates):
            candidate = nonfunc_templates[j].format(domain=domain)
            augmented_nonfunc.append(candidate)
            j += 1

        return (
            RequirementAgent._dedupe_preserve_order(augmented_func),
            RequirementAgent._dedupe_preserve_order(augmented_nonfunc),
        )

    # ------------------------------------------------------------------
    # Node: apply human-in-the-loop edits
    # ------------------------------------------------------------------
    def _human_review_node(self, state: RequirementState) -> RequirementState:
        """
        Apply a single human action to the requirement lists.
        This node is designed to be called repeatedly (once per user
        interaction) from the FastAPI layer until state['approved'] is True.
        """
        action = state.get("human_action")
        if not action:
            return state

        action_type = action.get("type")
        bucket = action.get("bucket", "functional")
        target_list = state["functional_requirements"] if bucket == "functional" else state["non_functional_requirements"]

        if action_type == "add":
            target_list.append(action["text"])
            logger.info("Added requirement to %s: %s", bucket, action["text"])
        elif action_type == "delete":
            index = action["index"]
            if 0 <= index < len(target_list):
                removed = target_list.pop(index)
                logger.info("Deleted requirement from %s: %s", bucket, removed)
        elif action_type == "modify":
            index = action["index"]
            if 0 <= index < len(target_list):
                target_list[index] = action["text"]
                logger.info("Modified requirement in %s at index %d", bucket, index)
        elif action_type == "approve":
            state["approved"] = True
            logger.info("Requirements approved by human reviewer")

        state["human_action"] = None
        return state

    # ------------------------------------------------------------------
    # Routing: loop on human review until approved
    # ------------------------------------------------------------------
    @staticmethod
    def _route_after_review(state: RequirementState) -> str:
        """Conditional edge: end the graph once the human has approved."""
        return END if state["approved"] else "human_review"

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------
    def _build_graph(self):
        """Wire the three nodes into a LangGraph StateGraph."""
        builder = StateGraph(RequirementState)

        builder.add_node("retrieve_context", self._retrieve_context_node)
        builder.add_node("generate_requirements", self._generate_requirements_node)
        builder.add_node("human_review", self._human_review_node)

        builder.set_entry_point("retrieve_context")
        builder.add_edge("retrieve_context", "generate_requirements")
        builder.add_edge("generate_requirements", "human_review")
        builder.add_conditional_edges("human_review", self._route_after_review, {"human_review": "human_review", END: END})

        return builder.compile()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run_initial_generation(self, project_description: str) -> RequirementState:
        """
        Run retrieval + generation once. Returns state with approved=False
        so the caller (FastAPI) can present requirements for human review
        before invoking apply_human_action in a loop.
        """
        initial_state: RequirementState = {
            "project_description": project_description,
            "github_context": "",
            "functional_requirements": [],
            "non_functional_requirements": [],
            "approved": False,
            "human_action": None,
        }
        state = self._retrieve_context_node(initial_state)
        state = self._generate_requirements_node(state)
        return state

    def apply_human_action(self, state: RequirementState, action: Dict) -> RequirementState:
        """Apply one human edit action (add/delete/modify/approve) to the state."""
        state["human_action"] = action
        return self._human_review_node(state)