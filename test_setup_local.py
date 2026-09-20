"""
Phase 0 Test Setup (Local CPU/GPU).

Runs a minimal end-to-end smoke check to verify your environment setup:
  - loads the smoke test problem from data/problems/smoke_test_problems.json
  - (optional) loads the local distilled model and runs one full injection episode
    for each of the 3 injection types
  - (optional) makes one API call to Gemini to confirm that path works too
  - writes transcripts to data/transcripts/smoke_test.jsonl

Usage:
    python test_setup_local.py                # runs both backends
    python test_setup_local.py --skip-api      # local model only
    python test_setup_local.py --skip-local    # API model only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
from src.harness import run_and_log_episode
from src.injections import InjectionType
from src.problems import load_smoke_test_problems


def main():
    parser = argparse.ArgumentParser(description="Phase 0 Test Setup (Local).")
    parser.add_argument("--skip-api", action="store_true", help="Skip the Gemini API backend check.")
    parser.add_argument("--skip-local", action="store_true", help="Skip the local model check.")
    args = parser.parse_args()

    problems = load_smoke_test_problems()
    if not problems:
        print("[ERROR] No smoke test problems found in data/problems/smoke_test_problems.json.")
        sys.exit(1)
    test_problem = problems[0]
    out_path = config.TRANSCRIPTS_DIR / "smoke_test.jsonl"

    if out_path.exists():
        out_path.unlink()

    print(f"Using smoke test problem: {test_problem.id} — {test_problem.prompt}")
    print(f"Writing transcript to: {out_path}\n")

    if not args.skip_local:
        print("=" * 70)
        print(f"LOCAL MODEL CHECK ({config.LOCAL_MODEL_ID} on {config.DEVICE})")
        print("=" * 70)
        from src.model_loader import LocalReasoningModel

        local_model = LocalReasoningModel()
        for injection_type in InjectionType:
            print(f"\n-- Running injection type: {injection_type.value} --")
            record = run_and_log_episode(
                model=local_model,
                problem=test_problem,
                injection_type=injection_type,
                out_path=out_path,
                fraction=config.DEFAULT_INJECTION_FRACTION,
            )
            print(f"Injected sentence: {record.injected_sentence}")
            print(f"Forced continuation (first 200 chars): {record.forced_continuation[:200]}")

    if not args.skip_api:
        print("\n" + "=" * 70)
        print(f"API MODEL CHECK (Gemini: {config.API_MODEL_ID})")
        print("=" * 70)
        if not config.GEMINI_API_KEY:
            print(
                "No GEMINI_API_KEY set — skipping API check. "
                "Set it in .env if you want to test this path."
            )
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

    print(f"\nDone. Read the full transcripts at: {out_path}")


if __name__ == "__main__":
    main()

