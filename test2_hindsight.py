#!/usr/bin/env python3
"""
Test 2: Hindsight "Just Ask"

Determines if the model can retroactively identify errors it failed to
spontaneously correct in Phase 1 by explicitly asking it.
"""
import argparse
from phase2_utils import load_transcripts, save_transcript
from src.modal_runner import ModalReasoningModel, app

def run_hindsight(input_transcript: str, output_file: str):
    # Load transcripts filtered for failed self-corrections
    records = load_transcripts(input_transcript, filter_categories=["Rationalization", "Commits to Error"])
    print(f"Loaded {len(records)} records for hindsight evaluation.")
    if not records:
        print("No eligible records found. Try running Phase 1 first or passing an unclassified transcript.")
        return

    model = ModalReasoningModel()
    
    hindsight_instruction = (
        "\n\nWas the previous reasoning step correct? "
        "Answer strictly Yes or No, and explain why.\nAnswer:"
    )
    
    for record in records:
        problem_id = record.get("problem_id", "unknown")
        inj_type = record.get("injection_type", "unknown")
        
        # Reconstruct the full text up to the continuation
        base_text = record.get("full_text", "")
        if not base_text:
            base_text = record.get("forced_prefix", "") + record.get("continuation", "")
            
        full_prompt = f"Below is a problem and a reasoning step:\n{base_text}{hindsight_instruction}"
        
        # Ask the model via standard text response (max 64 tokens, zero temperature)
        answer = model.generate_response(full_prompt).strip()
        
        # Basic heuristic parsing
        answer_lower = answer.lower()
        if answer_lower.startswith("yes"):
            hindsight_answer = "Yes"
        elif answer_lower.startswith("no"):
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
    # Defaulting to the phase 1 distilled output
    parser.add_argument("--input", default="data/transcripts/phase1_result_evaluate_adversarial_distilled.jsonl")
    parser.add_argument("--output", default="data/transcripts/phase2_hindsight.jsonl")
    args = parser.parse_args()
    
    # Run the Modal context block
    with app.run():
        run_hindsight(args.input, args.output)

