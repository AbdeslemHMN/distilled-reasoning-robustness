"""
The core injection harness.

Orchestrates the full episode for a single (problem, injection_type) pair:

    1. Generate a full CoT for the problem from a fresh model call.
    2. Split it into sentences and splice in an injected error at a chosen
       position (see injections.py).
    3. Feed the prefix (original CoT up to and including the injected
       sentence) back into the model and let it continue generating.
    4. Package everything into a structured record and append it to a
       .jsonl transcript file.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import config
from src.injections import InjectionType, build_forced_prefix_text, inject_error
from src.problems import Problem


class ReasoningModel(Protocol):
    """Structural type for backend models."""
    model_id: str
    
    # "true_token_prefill" for local models, "chat_assistant_prefill" for APIs.
    continuation_type: Literal["true_token_prefill", "chat_assistant_prefill"]

    # We now pass the fully formatted prompt from the harness, 
    # so the model wrapper just applies its chat template (if any).
    def generate_cot(self, formatted_prompt: str) -> str: ...
    def continue_from(self, forced_prefix_text: str) -> str: ...


@dataclass
class EpisodeRecord:
    """One row of the transcript log."""
    timestamp: float
    backend_model_id: str
    continuation_type: str        # Track how the backend handled the midpoint injection
    
    problem_id: str
    problem_prompt: str
    problem_reference_answer: str

    original_cot: str

    injection_type: str
    injection_fraction: float
    injection_index: int
    injected_sentence: str

    forced_prefix_cot: str        
    forced_continuation: str      
    full_forced_transcript: str   

    def to_dict(self) -> dict:
        return asdict(self)   


def format_problem_prompt(problem_prompt: str) -> str:
    """Single source of truth for base prompt text. Model wrappers should 
    NOT rewrite this; they should only wrap it in <|user|> / <|model|> tags."""
    return (
        f"Solve the following problem. Think step by step, "
        f"then give your final answer.\n\nProblem: {problem_prompt}\n\nSolution:\n"
    )


def run_episode(
    model: ReasoningModel,
    problem: Problem,
    injection_type: InjectionType,
    injection_fraction: float = config.DEFAULT_INJECTION_FRACTION,
) -> EpisodeRecord:
    """Run one full injection episode and return the structured record."""

    formatted_prompt = format_problem_prompt(problem.prompt)

    # Step 1: generate the original, uninterrupted CoT.
    # Notice we pass formatted_prompt now!
    original_cot = model.generate_cot(formatted_prompt)

    if not original_cot.strip():
        # Handle empty CoT prevention gracefully
        return EpisodeRecord(
            timestamp=time.time(),
            backend_model_id=model.model_id,
            continuation_type=model.continuation_type,
            problem_id=problem.id,
            problem_prompt=problem.prompt,
            problem_reference_answer=problem.answer,
            original_cot="",
            injection_type=injection_type.value,
            injection_fraction=injection_fraction,
            injection_index=-1,
            injected_sentence="N/A",
            forced_prefix_cot="N/A",
            forced_continuation="API Error / Empty CoT",
            full_forced_transcript="API Error / Empty CoT",
        )

    # Step 2: splice in the injected error.
    injection = inject_error(
        cot_text=original_cot,
        injection_type=injection_type,
        fraction=injection_fraction,
    )
    forced_prefix_text = build_forced_prefix_text(formatted_prompt, injection)

    # Step 3: regenerate the continuation from the forced prefix.
    forced_continuation = model.continue_from(forced_prefix_text)

    forced_prefix_cot = " ".join(injection.prefix_sentences)

    return EpisodeRecord(
        timestamp=time.time(),
        backend_model_id=model.model_id,
        continuation_type=model.continuation_type,
        problem_id=problem.id,
        problem_prompt=problem.prompt, # Keep original for clean data analysis
        problem_reference_answer=problem.answer,
        original_cot=original_cot,
        injection_type=injection.injection_type.value,
        injection_fraction=injection_fraction,
        injection_index=injection.injection_index,
        injected_sentence=injection.injected_sentence,
        forced_prefix_cot=forced_prefix_cot,
        forced_continuation=forced_continuation,
        full_forced_transcript=f"{forced_prefix_cot} {forced_continuation}",
    )


def append_record(record: EpisodeRecord, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def run_and_log_episode(
    model: ReasoningModel,
    problem: Problem,
    injection_type: InjectionType,
    out_path: Path,
    fraction: float = config.DEFAULT_INJECTION_FRACTION,
) -> EpisodeRecord:
    record = run_episode(model, problem, injection_type, fraction)
    append_record(record, out_path)
    return record


def run_batch(
    model: ReasoningModel,
    problems: list[Problem],
    injection_types: list[InjectionType],
    out_path: Path,
    injection_fraction: float = config.DEFAULT_INJECTION_FRACTION,
    verbose: bool = True,
) -> list[EpisodeRecord]:
    records = []
    total = len(problems) * len(injection_types)
    done = 0
    for problem in problems:
        for injection_type in injection_types:
            record = run_and_log_episode(
                model, problem, injection_type, out_path, injection_fraction
            )
            records.append(record)
            done += 1
            if verbose:
                print(
                    f"[{done}/{total}] {problem.id} x {injection_type.value} -> "
                    f"logged to {out_path.name}"
                )
    return records

def prepare_problem_prompt(problem: Problem | dict | str) -> str:
    """Formats the base prompt from a Problem object, dictionary, or raw string."""
    if isinstance(problem, str):
        prompt_text = problem
    elif isinstance(problem, dict):
        prompt_text = problem.get("prompt", problem.get("problem", ""))
    else:
        prompt_text = getattr(problem, "prompt", str(problem))
    return format_problem_prompt(prompt_text)
