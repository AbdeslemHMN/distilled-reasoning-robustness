"""
Phase 0 Test Setup (Modal Cloud GPU).

Runs a minimal end-to-end smoke check on a remote Modal GPU (T4):
  - loads the smoke test problem from data/problems/smoke_test_problems.json
  - runs error-injection on Modal Cloud T4 GPU
  - (optional) tests Gemini API backend
  - logs output to data/transcripts/smoke_test.jsonl

Usage:
    modal run test_setup_modal.py                  # runs local harness + remote T4 GPU
    modal run test_setup_modal.py --skip-api       # skips Gemini API check
"""

from __future__ import annotations

import modal
import config
from src.modal_runner import app, DistilledModelRunner, ModalReasoningModel

if modal.is_local():
    from src.harness import run_and_log_episode
    from src.injections import InjectionType
    from src.problems import load_smoke_test_problems


@app.local_entrypoint()
def main(skip_api: bool = False):
    print("=" * 70)
    print(f"PHASE 0 TEST SETUP VIA MODAL (Remote GPU: T4, Model: {config.LOCAL_MODEL_ID})")
    print("=" * 70)

    problems = load_smoke_test_problems()
    if not problems:
        print("[ERROR] No smoke test problems found in data/problems/smoke_test_problems.json.")
        return

    test_problem = problems[0]
    out_path = config.TRANSCRIPTS_DIR / "smoke_test.jsonl"

    if out_path.exists():
        out_path.unlink()

    print(f"Using smoke test problem: {test_problem.id} — {test_problem.prompt}")
    print(f"Local transcript path: {out_path}\n")

    # Connect to the remote GPU runner
    runner = DistilledModelRunner()
    model = ModalReasoningModel(runner)

    # Run the 3 error injection episodes
    for injection_type in InjectionType:
        print(f"\n-- Running injection type: {injection_type.value} --")
        record = run_and_log_episode(
            model=model,
            problem=test_problem,
            injection_type=injection_type,
            out_path=out_path,
            fraction=config.DEFAULT_INJECTION_FRACTION,
        )
        print(f"Injected sentence: {record.injected_sentence}")
        print(f"Forced continuation (first 200 chars): {record.forced_continuation[:200]}")

    # Optional Gemini API check
    if not skip_api:
        print("\n" + "=" * 70)
        print(f"API MODEL CHECK (Gemini: {config.API_MODEL_ID})")
        print("=" * 70)
        if not config.GEMINI_API_KEY:
            print("No GEMINI_API_KEY set — skipping API check.")
        else:
            from src.api_client import APIReasoningModel

            api_model = APIReasoningModel()
            record = run_and_log_episode(
                model=api_model,
                problem=test_problem,
                injection_type=InjectionType.OBVIOUS_ARITHMETIC,
                out_path=out_path,
                fraction=config.DEFAULT_INJECTION_FRACTION,
            )
            print(f"Injected sentence: {record.injected_sentence}")
            print(f"Forced continuation (first 200 chars): {record.forced_continuation[:200]}")

    print(f"\nDone! Successfully logged transcripts to: {out_path}")

