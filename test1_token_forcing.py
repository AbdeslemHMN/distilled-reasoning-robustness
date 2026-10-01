#!/usr/bin/env python3
"""
Test 1: Token-Forcing Baseline

Tests if self-correction is a fragile surface behavior by suppressing
common correction reflex tokens using the Modal GPU Runner.
"""
import argparse
from phase2_utils import load_problems, save_transcript
from src.harness import format_problem_prompt
from src.injections import inject_error, InjectionType, build_forced_prefix_text

def run_token_forcing(problems_file: str, output_file: str, use_local: bool = False):
    target_words = ["Wait", "wait", "Actually", "however", "But", "Correction"]
    
    if use_local:
        from src.model_loader import LocalReasoningModel
        model = LocalReasoningModel()
        suppressed_ids = model.get_suppressed_token_ids(target_words)
    else:
        from src.modal_runner import ModalReasoningModel
        model = ModalReasoningModel()
        print(f"Retrieving suppressed token IDs for words: {target_words}")
        suppressed_ids = model.runner.get_suppressed_token_ids.remote(target_words)
    print(f"Suppressed Token IDs mapped: {suppressed_ids}")
    
    problems = load_problems(problems_file)
    print(f"Loaded {len(problems)} problems.")
    
    injection_types = [
        InjectionType.OBVIOUS_ARITHMETIC,
        InjectionType.PLAUSIBLE_LOGICAL,
        InjectionType.SUBTLE_NUMERICAL
    ]
    
    for p in problems:
        problem_id = p["id"]
        base_prompt = format_problem_prompt(p.get("prompt", p.get("problem", "")))
        
        # 1. Generate base CoT to find a natural injection point
        base_cot = model.generate_cot(base_prompt)
        
        for inj_type in injection_types:
            try:
                # 2. Inject at 30% through the CoT
                injection = inject_error(base_cot, inj_type, fraction=0.3)
                
                # Check for static injections in sample_problems.json
                if "injections" in p and inj_type.value in p["injections"]:
                    custom_injection = p["injections"][inj_type.value]
                    injection.injected_sentence = custom_injection
                    injection.prefix_sentences[-1] = custom_injection
                
                forced_prefix = build_forced_prefix_text(base_prompt, injection)
                
                # 3. Generate continuation WITH token suppression via remote call
                continuation = model.continue_from(forced_prefix, suppress_tokens=suppressed_ids)
                
                record = {
                    "problem_id": problem_id,
                    "injection_type": inj_type.value,
                    "forced_prefix": forced_prefix,
                    "continuation": continuation,
                    "full_text": forced_prefix + continuation,
                    "evaluation_category": "TODO_EVALUATE_MANUALLY" 
                }
                save_transcript(output_file, record)
                print(f"Processed {problem_id} - {inj_type.value} (Token Forcing)")
                
            except Exception as e:
                print(f"Error processing {problem_id} with {inj_type.value}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="data/problems/sample_problems.json")
    parser.add_argument("--output", default="data/transcripts/phase2_token_forcing.jsonl")
    parser.add_argument("--local", action="store_true", help="Run locally")
    args = parser.parse_args()
    
    if args.local:
        run_token_forcing(args.problems, args.output, use_local=True)
    else:
        from src.modal_runner import app
        with app.run():
            run_token_forcing(args.problems, args.output, use_local=False)

