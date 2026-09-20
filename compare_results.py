import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Compare Phase 1 Adversarial Reasoning Transcripts.")
    parser.add_argument('--distilled_file', default='data/transcripts/phase1_result_evaluate_adversarial_distilled.jsonl', help="Path to distilled JSONL file")
    parser.add_argument('--rl_file', default='data/transcripts/phase1_result_evaluate_adversarial_rl.jsonl', help="Path to RL-native (Gemini) JSONL file")
    parser.add_argument('--output', default='data/transcripts/phase1_comparison_table.md', help="Path to save output markdown table")
    parser.add_argument('--percentages', action='store_true', help="Compute and print behavioral percentages summary")
    args = parser.parse_args()

    # Read distilled file
    distilled_data = {}
    if Path(args.distilled_file).exists():
        with open(args.distilled_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                problem_id = record.get('problem_id')
                injection_type = record.get('injection_type')
                category = record.get('evaluation_category', 'N/A')
                
                if problem_id not in distilled_data:
                    distilled_data[problem_id] = {}
                distilled_data[problem_id][injection_type] = category
    else:
        print(f"[WARNING] Distilled file not found: {args.distilled_file}")

    # Read RL file
    rl_data = {}
    if Path(args.rl_file).exists():
        with open(args.rl_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                problem_id = record.get('problem_id')
                injection_type = record.get('injection_type')
                category = record.get('evaluation_category', 'N/A')
                
                if problem_id not in rl_data:
                    rl_data[problem_id] = {}
                rl_data[problem_id][injection_type] = category
    else:
        print(f"[WARNING] RL file not found: {args.rl_file}")

    # Merge and build table
    all_problems = sorted(list(set(list(distilled_data.keys()) + list(rl_data.keys()))))
    
    table_lines = [
        "| Problem | Injection Type | Distilled Model Behavior | API Target Model Behavior |",
        "| :--- | :--- | :--- | :--- |"
    ]
    
    for problem_id in all_problems:
        # Collect all injection types for this problem
        dist_inj = distilled_data.get(problem_id, {})
        rl_inj = rl_data.get(problem_id, {})
        all_inj_types = sorted(list(set(list(dist_inj.keys()) + list(rl_inj.keys()))))
        
        for inj_type in all_inj_types:
            dist_cat = dist_inj.get(inj_type, "N/A")
            rl_cat = rl_inj.get(inj_type, "N/A")
            table_lines.append(f"| {problem_id} | `{inj_type}` | {dist_cat} | {rl_cat} |")

    table_content = "\n".join(table_lines) + "\n"
    print(table_content)
    
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(table_content)
        print(f"\n[INFO] Saved comparison table to: {args.output}")

    if args.percentages:
        try:
            from calculate_phase1_percentages import parse_comparison_table, compute_statistics, print_summary
            rows = parse_comparison_table(Path(args.output))
            stats = compute_statistics(rows)
            print_summary(stats)
        except Exception as e:
            print(f"[WARNING] Could not compute percentages: {e}")

if __name__ == "__main__":
    main()

