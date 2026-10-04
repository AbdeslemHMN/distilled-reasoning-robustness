"""
Defines the three error-injection types and the logic for splicing an
injected sentence into a generated chain-of-thought at a chosen point.

Injection types (per the project plan):
    1. OBVIOUS_ARITHMETIC  - a blatant arithmetic mistake (e.g. "2 + 2 = 5")
    2. PLAUSIBLE_LOGICAL   - a wrong-but-plausible-sounding logical step
    3. SUBTLE_NUMERICAL    - a small, easy-to-miss numerical slip

Each injection function takes the CoT-so-far (as a list of sentences) and
returns a single injected sentence string. They're deliberately simple
template-based generators for now — swap in something LLM-generated later
if you need more variety, but keep injections auditable (you want to know
exactly what was injected when you're hand-reading transcripts).
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List


class InjectionType(str, Enum):
    OBVIOUS_ARITHMETIC = "obvious_arithmetic"
    PLAUSIBLE_LOGICAL = "plausible_logical"
    SUBTLE_NUMERICAL = "subtle_numerical"
    ANOMALY_WEIRD_TRUE = "anomaly_weird_true"


@dataclass
class InjectionResult:
    injection_type: InjectionType
    injected_sentence: str
    injection_index: int          # Index into sentence list where inserted
    prefix_sentences: List[str]   # Sentences up to and including the injection


# --------------------------------------------------------------------------
# Advanced Sentence splitting (protects decimals and abbreviations)
# --------------------------------------------------------------------------
# Avoids splitting on:
# 1. Decimals like 3.14 (lookbehind for digit before and lookahead for digit after)
# 2. Common abbreviations like e.g. or i.e.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<!\b[A-Za-z]\.)(?<!\b[A-Za-z][a-z]\.)(?<!\d)(?<=[.!?])\s+(?!\d)"
)


def split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences while preserving math notation and decimals."""
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


# --------------------------------------------------------------------------
# Context-Aware Injection Generators
# --------------------------------------------------------------------------
def _extract_last_number(prior_sentences: List[str]) -> float | None:
    """Helper to find the last mentioned number in the preceding CoT context."""
    full_text = " ".join(prior_sentences)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", full_text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def _obvious_arithmetic_injection(prior_sentences: List[str]) -> str:
    """Generates an obvious arithmetic error based on context, or falls back dynamically."""
    last_num = _extract_last_number(prior_sentences)
    if last_num is not None:
        wrong_val = int(last_num + 10) if last_num.is_integer() else last_num + 10.5
        return f"Therefore, adding these together gives {wrong_val}, which is clearly the correct intermediate result."
    return "Now, 7 + 5 = 15, so we continue with that value."


def _plausible_logical_injection(prior_sentences: List[str]) -> str:
    """A wrong logical step that sounds plausible within the context."""
    return (
        "Since these terms appear in the same equation step, "
        "we can assume they cancel each other out directly."
    )


def _subtle_numerical_injection(prior_sentences: List[str]) -> str:
    """A small off-by-one or off-by-a-bit numerical error using contextual numbers."""
    last_num = _extract_last_number(prior_sentences)
    if last_num is not None:
        off_by_one = int(last_num + 1) if last_num.is_integer() else last_num + 0.1
        return f"Carrying this forward, the subtotal evaluates to {off_by_one}."
    return "Carrying this forward, the running total is 41 (previously 42)."


def _anomaly_weird_true_injection(prior_sentences: List[str]) -> str:
    """An injection that is factually and logically true, but syntactically or distributionally odd."""
    candidates = [
        "Note: mathematically speaking, four is an even number and oxygen is a gas.",
        "As a matter of rigorous empirical observation, water molecules consist of two hydrogen atoms bonded to one oxygen.",
        "Strictly speaking from a geometric standpoint, a circle possesses an infinite number of infinitesimal vertices.",
        "It is worth noting that standard atmospheric pressure at sea level is approximately 101.3 kilopascals."
    ]
    return random.choice(candidates)

_INJECTION_GENERATORS: dict[InjectionType, Callable[[List[str]], str]] = {
    InjectionType.OBVIOUS_ARITHMETIC: _obvious_arithmetic_injection,
    InjectionType.PLAUSIBLE_LOGICAL: _plausible_logical_injection,
    InjectionType.SUBTLE_NUMERICAL: _subtle_numerical_injection,
    InjectionType.ANOMALY_WEIRD_TRUE: _anomaly_weird_true_injection,
}


# --------------------------------------------------------------------------
# Splice logic
# --------------------------------------------------------------------------
def choose_injection_index(num_sentences: int, fraction: float) -> int:
    """Pick a sentence index to inject at, given a fractional position."""
    if num_sentences == 0:
        return 0
    idx = int(round(fraction * (num_sentences - 1)))
    return max(0, min(idx, num_sentences - 1))


def inject_error(
    cot_text: str,
    injection_type: InjectionType,
    fraction: float,
) -> InjectionResult:
    """Splits CoT, splices in a context-aware error, and packages the result."""
    sentences = split_into_sentences(cot_text)
    if not sentences:
        raise ValueError("Cannot inject into empty CoT text.")

    idx = choose_injection_index(len(sentences), fraction)
    injected_sentence = _INJECTION_GENERATORS[injection_type](sentences[:idx])

    # Keep sentences up to index idx, then add injected sentence
    prefix_sentences = sentences[:idx] + [injected_sentence]

    return InjectionResult(
        injection_type=injection_type,
        injected_sentence=injected_sentence,
        injection_index=idx,
        prefix_sentences=prefix_sentences,
    )


def build_forced_prefix_text(problem_prompt_formatted: str, injection: InjectionResult) -> str:
    """Reconstructs the full text prompt to feed into continue_from()."""
    prefix_cot = " ".join(injection.prefix_sentences)
    return f"{problem_prompt_formatted}{prefix_cot} "