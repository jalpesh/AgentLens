from __future__ import annotations

from . import rollups
from .detectors import DETECTORS, Finding, run_all, run_session
from .playbook import CURATED_REFERENCE, Rule, build_playbook, render as render_playbook
from .prompt_score import (
    PromptReport,
    content_type_budget,
    count_tokens,
    prompt_trends,
    score_across_models,
    score_prompt,
)
from .fingerprint import ProjectProfile, fingerprint, fingerprint_all
from .rewrite import Scaffold, detect_test_command, llm_available, llm_rewrite, scaffold
from .session_detail import build_session_detail
from .skills import Artifact, Suggestion, generic_suggestions, suggest

__all__ = [
    "rollups",
    "DETECTORS",
    "Finding",
    "run_all",
    "run_session",
    "PromptReport",
    "count_tokens",
    "score_prompt",
    "content_type_budget",
    "score_across_models",
    "prompt_trends",
    "Rule",
    "build_playbook",
    "render_playbook",
    "CURATED_REFERENCE",
    "Scaffold",
    "scaffold",
    "detect_test_command",
    "llm_available",
    "llm_rewrite",
    "ProjectProfile",
    "fingerprint",
    "fingerprint_all",
    "build_session_detail",
    "Artifact",
    "Suggestion",
    "suggest",
    "generic_suggestions",
]
