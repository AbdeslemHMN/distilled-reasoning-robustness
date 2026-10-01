#!/usr/bin/env python3
"""
Test 3: Anomaly Control

Tests whether the model corrects logical wrongness or merely distributional
oddity by injecting weird-but-true statements.
"""
import argparse
from phase2_utils import load_problems, save_transcript
from src.modal_runner import ModalReasoningModel, app
from src.harness import format_problem_prompt
from src.injections import inject_error, InjectionType, build_forced_prefix_text

def run_anomaly(problems_file: str, output_file: str):
    
    model = ModalReasoningModel()
    
    problems = load_problems(problems_file)
    print(f"Loaded {len(problems)} problems for anomaly test.")
    
    # We test the anomaly injection
    inj_type = InjectionType.ANOMALY_WEIRD_TRUE
    
    for p in problems:
        problem_id = p["id"]
        base_prompt = format_problem_prompt(p.get("prompt", p.get("problem", "")))
        
        # 1. Generate a baseline short CoT to determine the injection point.
        base_cot = model.generate_cot(base_prompt)
        
        try:
            # Inject at the same 30% mark as Phase 1
            injection = inject_error(base_cot, inj_type, fraction=0.3)
            
            # If the user put static injections in sample_problems.json, respect that
            if "injections" in p and inj_type.value in p["injections"]:
                custom_injection = p["injections"][inj_type.value]
                # Overwrite the injected sentence with the static one
                injection.injected_sentence = custom_injection
                injection.prefix_sentences[-1] = custom_injection
            
            forced_prefix = build_forced_prefix_text(base_prompt, injection)
            
            # 2. Generate continuation without any token suppression
            continuation = model.continue_from(forced_prefix)
            
            record = {
                "problem_id": problem_id,
                "injection_type": inj_type.value,
                "forced_prefix": forced_prefix,
                "continuation": continuation,
                "full_text": forced_prefix + continuation,
                "evaluation_category": "TODO_EVALUATE_MANUALLY" 
            }
            save_transcript(output_file, record)
            print(f"Processed {problem_id} - Anomaly Control")
            
        except Exception as e:
            print(f"Error processing {problem_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="data/problems/sample_problems.json")
    parser.add_argument("--output", default="data/transcripts/phase2_anomaly.jsonl")
    args = parser.parse_args()
    
    # Run the Modal context block
    with app.run():
        run_anomaly(args.problems, args.output)

