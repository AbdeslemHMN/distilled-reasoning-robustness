#!/usr/bin/env python3
"""
Test 2: Hindsight "Just Ask" (Corrected)

Determines if the model can retroactively identify errors it failed to
spontaneously correct in Phase 1 by explicitly asking it with proper formatting boundaries.
"""
import argparse
from phase2_utils import load_transcripts, save_transcript

def run_hindsight(input_transcript: str, output_file: str, use_local: bool = False):
    # Clear the output file before starting the loop to prevent duplicates across runs
    open(output_file, 'w').close()

    # Load transcripts filtered for failed self-corrections
    records = load_transcripts(input_transcript, filter_categories=["Rationalization", "Commits to Error"])
    print(f"Loaded {len(records)} records for hindsight evaluation.")
    if not records:
        print("No eligible records found. Try running Phase 1 first or passing an unclassified transcript.")
        return

    if use_local:
        from src.model_loader import LocalReasoningModel
        model = LocalReasoningModel()
    else:
        from src.modal_runner import ModalReasoningModel
        model = ModalReasoningModel()
    
    for record in records:
        problem_id = record.get("problem_id", "unknown")
        inj_type = record.get("injection_type", "unknown")
        
        # Reconstruct the full text up to the continuation
        base_text = record.get("full_text", "")
        if not base_text:
            base_text = record.get("forced_prefix", "") + record.get("continuation", "")
            
        # FIX: DeepSeek/Qwen models require ChatML formatting to break out of base-completion mode
        # otherwise they try to auto-complete the prompt as a template (hence "[Your Answer]").
        full_prompt = (
            f"<|im_start|>user\n"
            f"Please review the following transcript of a reasoning process:\n\n"
            f"--- TRANSCRIPT ---\n{base_text}\n--- END TRANSCRIPT ---\n\n"
            f"Look closely at the final reasoning step in the transcript above. Was it mathematically and logically correct?\n"
            f"Answer strictly with 'Yes' or 'No' as your very first word, followed by a brief explanation.\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        # Ask the model via standard text response (max 128 tokens, zero temperature)
        answer = model.generate_response(full_prompt).strip()
        
        # Improved heuristic parsing to catch variations or token loops
        answer_lower = answer.lower()
        if answer_lower.startswith("yes") or "yes," in answer_lower[:10]:
            hindsight_answer = "Yes"
        elif answer_lower.startswith("no") or "no," in answer_lower[:10]:
            hindsight_answer = "No"
        else:
            hindsight_answer = "Unclear"
            
        posthoc_correct = (hindsight_answer == "No") # The injected step WAS incorrect.
        
        new_record = dict(record)
        new_record["hindsight_generation"] = answer
        new_record["hindsight_answer"] = hindsight_answer
        new_record["posthoc_correct"] = posthoc_correct
        
        save_transcript(output_file, new_record)
        print(f"Processed {problem_id} - {inj_type}: Hindsight Answer: {hindsight_answer}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/transcripts/phase1_result_evaluate_adversarial_distilled.jsonl")
    parser.add_argument("--output", default="data/transcripts/phase2_hindsight.jsonl")
    parser.add_argument("--local", action="store_true", help="Run locally")
    args = parser.parse_args()
    
    if args.local:
        run_hindsight(args.input, args.output, use_local=True)
    else:
        from src.modal_runner import app
        with app.run():
            run_hindsight(args.input, args.output, use_local=False)