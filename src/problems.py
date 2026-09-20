"""
Loads the bank of math/logic problems used for injection experiments.

Problems live in:
  - data/problems/smoke_test_problems.json (small 5-problem set for test setup / smoke testing)
  - data/problems/sample_problems.json (15-20 problems for Phase 1 evaluations)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import config


@dataclass
class Problem:
    id: str
    prompt: str
    answer: str
    category: str = "general"

    @property
    def problem(self) -> str:
        """Alias for prompt."""
        return self.prompt

    @property
    def ground_truth(self) -> str:
        """Alias for answer."""
        return self.answer


def load_problems(path: Union[Path, str, None] = None) -> List[Problem]:
    """Load problems from JSON, supporting either schema (prompt/problem, answer/ground_truth)."""
    p_path = Path(path) if path else config.PROBLEMS_PATH
    with open(p_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    problems = []
    for p in raw:
        prompt = p.get("prompt") or p.get("problem", "")
        answer = p.get("answer") or p.get("ground_truth", "")
        category = p.get("category", "general")
        problems.append(
            Problem(
                id=p["id"],
                prompt=prompt,
                answer=answer,
                category=category,
            )
        )
    return problems


def load_smoke_test_problems() -> List[Problem]:
    """Load the dedicated smoke test problems for test setup verification."""
    return load_problems(config.SMOKE_TEST_PROBLEMS_PATH)
