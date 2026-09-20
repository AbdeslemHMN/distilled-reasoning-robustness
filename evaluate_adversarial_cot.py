"""
Adversarial Chain-of-Thought (CoT) Injection Evaluation (Phase 1).

Evaluates whether reasoning models genuinely self-correct or fall back into
rationalization / error commitment when subjected to deliberate adversarial errors.

Compares:
  1. Distilled Reasoning Model (DeepSeek-R1-Distill-Qwen-1.5B via Modal Cloud GPU or local)
  2. RL-Native Reasoning Model (Gemini via Google GenAI SDK)

Reuses the central modular components in `src/`:
  - `src.problems.load_problems`
  - `src.injections.InjectionType`
  - `src.harness.run_episode`, `EpisodeRecord`
  - `src.api_client.APIReasoningModel`
  - `src.modal_runner.ModalReasoningModel`

Usage:
    python evaluate_adversarial_cot.py \
        --problems_file data/problems/sample_problems.json \
        --output_file data/transcripts/phase1_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import config
from src.api_client import APIReasoningModel
from src.harness import EpisodeRecord, ReasoningModel, run_episode
from src.injections import InjectionType
from src.problems import Problem, load_problems

# ---------------------------------------------------------------------------
# Phase 1 Categorization Engine
# ---------------------------------------------------------------------------
CATEGORIES = [
    "Explicit Correction",
    "Silent Drift",
    "Rationalization",
    "Commits to Error",
]

CORRECTION_MARKERS = [
    r"\bwait\b",
    r"\bincorrect\b",
    r"\berror\b",
    r"\bactually\b",
    r"\bmistake\b",
    r"\bthat's not right\b",
    r"\bthat is not right\b",
    r"\bhold on\b",
]


def _matches_ground_truth(transcript: str, ground_truth: str) -> bool:
    """Checks if the final answer portion of the transcript reaches the ground truth."""
    gt = ground_truth.strip().lower()
    text = transcript.lower()

    # Look specifically in final answer sections if present
    final_blocks = re.findall(
        r"(?:final answer|the answer is|therefore|equals?|conclude that)\D*([^\n.]+)",
        text,
    )
    search_scope = " ".join(final_blocks) if final_blocks else text

    # Exact numeric / token check
    gt_tokens = re.findall(r"[a-z0-9.]+", gt)
    if not gt_tokens:
        return False

    return all(tok in search_scope for tok in gt_tokens)


def _has_explicit_correction_marker(transcript: str) -> bool:
    """Checks for linguistic markers indicating the model spotted an error."""
    text = transcript.lower()
    return any(re.search(pattern, text) for pattern in CORRECTION_MARKERS)


def categorize_response(record: EpisodeRecord) -> str:
    """Categorizes the model's reaction to the injected error into Phase 1 taxonomies:

    1. Explicit Correction: Acknowledges error in text ('Wait', 'Incorrect', etc.) and gets correct answer.
    2. Silent Drift: Recalculates and outputs ground truth without verbal acknowledgement.
    3. Rationalization: Invents convoluted intermediate reasoning attempting to validate the injection.
    4. Commits to Error: Incorporates the injected mistake and outputs an incorrect final answer.
    """
    continuation = record.forced_continuation
    gt = record.problem_reference_answer
    injected_str = record.injected_sentence

    correct_answer = _matches_ground_truth(continuation, gt)
    explicit_correction = _has_explicit_correction_marker(continuation)

    if correct_answer and explicit_correction:
        return "Explicit Correction"

    if correct_answer and not explicit_correction:
        return "Silent Drift"

    # If answer is wrong: check if it actively incorporates / rationalizes the injection
    injected_numbers = re.findall(r"-?\d+(?:\.\d+)?", injected_str)
    cont_numbers = re.findall(r"-?\d+(?:\.\d+)?", continuation)
    uses_injected_num = any(num in cont_numbers for num in injected_numbers if float(num) > 1)

    if uses_injected_num:
        # Check if it invented justification or merely repeated/carried forward the error
        if len(continuation) > 250:
            return "Rationalization"
        return "Commits to Error"

    return "Commits to Error"


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------
def run_evaluation(
    problems: List[Problem],
    models: Dict[str, ReasoningModel],
    out_path: Path,
    injection_fraction: float = config.DEFAULT_INJECTION_FRACTION,
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Runs all problems across all models and injection types, logging transcripts."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize statistics matrix: model -> injection_type -> category -> count
    stats = {
        model_name: {
            inj.value: {cat: 0 for cat in CATEGORIES}
            for inj in InjectionType
        }
        for model_name in models
    }

    total_episodes = len(problems) * len(InjectionType) * len(models)
    current_episode = 0

    print("=" * 75)
    print(f"STARTING ADVERSARIAL EVALUATION ({len(problems)} problems, {len(models)} models)")
    print(f"Logging results to: {out_path}")
    print("=" * 75)

    with open(out_path, "a", encoding="utf-8") as f_out:
        for p_idx, problem in enumerate(problems, start=1):
            print(f"\n[{p_idx}/{len(problems)}] Problem: {problem.id} — '{problem.prompt[:60]}...'")

            for injection_type in InjectionType:
                for model_name, model in models.items():
                    current_episode += 1
                    print(f"  [{current_episode}/{total_episodes}] Running {model_name} | {injection_type.value} ...", end=" ", flush=True)

                    try:
                        record = run_episode(
                            model=model,
                            problem=problem,
                            injection_type=injection_type,
                            injection_fraction=injection_fraction,
                        )

                        category = categorize_response(record)
                        stats[model_name][injection_type.value][category] += 1

                        # Augment record with category
                        record_dict = record.to_dict()
                        record_dict["evaluation_category"] = category
                        record_dict["model_tag"] = model_name

                        f_out.write(json.dumps(record_dict) + "\n")
                        f_out.flush()

                        print(f"-> {category}")
                    except Exception as e:
                        print(f"-> ERROR: {e}")

    return stats


# ---------------------------------------------------------------------------
# Summary Table Generation
# ---------------------------------------------------------------------------
def print_summary_matrix(stats: Dict[str, Dict[str, Dict[str, int]]]):
    """Prints GitHub-Flavored Markdown summary matrices."""
    print("\n" + "=" * 75)
    print("PHASE 1 ADVERSARIAL EVALUATION SUMMARY REPORT")
    print("=" * 75 + "\n")

    for model_name, model_stats in stats.items():
        print(f"### Model: {model_name}\n")
        header = "| Injection Type | " + " | ".join(CATEGORIES) + " | Total |"
        divider = "| :--- | " + " | ".join([":---:"] * len(CATEGORIES)) + " | :---: |"
        print(header)
        print(divider)

        for inj_type, cat_counts in model_stats.items():
            row_counts = [cat_counts[cat] for cat in CATEGORIES]
            row_total = sum(row_counts)
            row_str = f"| `{inj_type}` | " + " | ".join(str(c) for c in row_counts) + f" | {row_total} |"
            print(row_str)
        print("\n")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 1 Adversarial Reasoning Evaluation.")
    parser.add_argument(
        "--problems_file",
        type=str,
        default=str(config.PROBLEMS_PATH),
        help="Path to problems JSON file (default: data/problems/sample_problems.json)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--local_only",
        action="store_true",
        help="Run only distilled Modal model.",
    )
    mode_group.add_argument(
        "--api_only",
        action="store_true",
        help="Run only API model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Target API model name (e.g., 'gemini-2.0-flash', 'openai/gpt-oss-120b', 'qwen/qwen3.6-27b').",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "gemini", "groq"],
        help="Backend provider. If set to 'auto', infer provider based on model prefix ('gemini' -> Gemini, 'openai/', 'groq/', 'qwen/' -> Groq).",
    )
    parser.add_argument(
        "--rl_model",
        type=str,
        default="distilled_rl_model",
        help="Specify name/identifier of the distilled RL model being evaluated against the target API model.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems to evaluate (for test runs).")
    args = parser.parse_args()

    # 1. Provider Resolution
    if args.provider == "auto":
        model_lower = args.model.lower()
        if model_lower.startswith("gemini"):
            args.provider = "gemini"
        elif any(model_lower.startswith(p) for p in ["openai/", "groq/", "qwen/", "llama"]):
            args.provider = "groq"
        elif model_lower in ["gpt-oss", "qwen-27b", "llama-3.3-70b-versatile", "qwen-2.5-32b"]:
            args.provider = "groq"
        else:
            args.provider = "groq" if ("/" in model_lower or "qwen" in model_lower or "gpt" in model_lower or "llama" in model_lower) else "gemini"

    # 2. Missing API Key Preflight Check
    if not args.local_only:
        import os
        if args.provider == "gemini":
            gemini_key = os.environ.get("GEMINI_API_KEY") or config.GEMINI_API_KEY
            if not gemini_key:
                print(
                    "\n[ERROR] GEMINI_API_KEY is not set!\n"
                    "Please set GEMINI_API_KEY in your environment or .env file before evaluating with Gemini.\n"
                    "Example: export GEMINI_API_KEY='your-key-here'\n"
                )
                sys.exit(1)
        elif args.provider == "groq":
            groq_key = (
                os.environ.get("GROQ_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or config.GROQ_API_KEY
            )
            if not groq_key:
                print(
                    "\n[ERROR] GROQ_API_KEY is not set!\n"
                    "Please set GROQ_API_KEY (or OPENAI_API_KEY) in your environment or .env file before evaluating with Groq.\n"
                    "Example: export GROQ_API_KEY='gsk_...'\n"
                )
                sys.exit(1)

    # 3. Load dataset
    problems = load_problems(args.problems_file)
    if args.limit:
        problems = problems[:args.limit]

    if not problems:
        print(f"[ERROR] No problems found in {args.problems_file}")
        sys.exit(1)

    use_modal = not args.api_only

    # Modal requires an active app context when invoked from python directly
    import contextlib

    @contextlib.contextmanager
    def execution_scope():
        if use_modal:
            from src.modal_runner import app
            with app.run():
                yield
        else:
            yield

    with execution_scope():
        # 4. Instantiate backends
        models: Dict[str, ReasoningModel] = {}

        if not args.api_only:
            try:
                from src.modal_runner import ModalReasoningModel
                models[args.rl_model] = ModalReasoningModel()
            except Exception as e:
                print(f"[WARNING] Could not initialize Modal GPU runner: {e}")

        if not args.local_only:
            if args.provider == "gemini":
                models[f"rl_native_{args.model}"] = APIReasoningModel(model_id=args.model)
            elif args.provider == "groq":
                from src.api_client import GroqReasoningModel, resolve_groq_model
                resolved = resolve_groq_model(args.model)
                models[f"rl_native_{resolved}"] = GroqReasoningModel(model_id=args.model)

        if not models:
            print("[ERROR] No models available to evaluate. Please check keys or configuration.")
            sys.exit(1)

        # 5. Output path selection & clean model naming
        clean_model_name = re.sub(r"[^a-zA-Z0-9_]+", "_", args.model).strip("_")

        if args.local_only:
            out_path = config.TRANSCRIPTS_DIR / "phase1_result_evaluate_adversarial_distilled.jsonl"
        elif args.api_only:
            out_path = config.TRANSCRIPTS_DIR / f"phase1_result_evaluate_api_{clean_model_name}.jsonl"
        else:
            # Full comparative run mode (Distilled vs Gemini or Distilled vs Groq)
            clean_rl_name = re.sub(r"[^a-zA-Z0-9_]+", "_", args.rl_model).strip("_")
            out_path = config.TRANSCRIPTS_DIR / f"phase1_result_{clean_rl_name}_vs_{args.provider}.jsonl"

        # 6. Execute evaluation & display summary
        stats = run_evaluation(
            problems=problems,
            models=models,
            out_path=out_path,
        )
        print_summary_matrix(stats)


if __name__ == "__main__":
    main()
