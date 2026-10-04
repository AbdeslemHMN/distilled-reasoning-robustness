import json
import argparse
import os
from collections import defaultdict

def load_jsonl(filepath):
    data = []
    if not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}")
        return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def calc_category_distribution(data, filter_injection=None):
    dist = defaultdict(int)
    total = 0
    for row in data:
        if filter_injection and row.get('injection_type') != filter_injection:
            continue
        cat = row.get('evaluation_category', 'Unknown')
        # Standardize labels just in case
        if cat.lower() == 'silent drift': cat = 'Silent Drift'
        dist[cat] += 1
        total += 1
    
    pcts = {}
    if total > 0:
        for k, v in dist.items():
            pcts[k] = (v / total) * 100
    return pcts, total

def analyze_suite(p1_path, p2_t1_path, p2_t2_path, p2_t3_path, output_md):
    p1_data = load_jsonl(p1_path)
    t1_data = load_jsonl(p2_t1_path)
    t2_data = load_jsonl(p2_t2_path)
    t3_data = load_jsonl(p2_t3_path)

    cats = ["Explicit Correction", "Silent Drift", "Rationalization", "Commits to Error"]
    injection_types = ["obvious_arithmetic", "plausible_logical", "subtle_numerical"]

    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# Phase 2 Suite Analysis Report\n\n")
        
        # ---------------------------------------------------------
        # 1. Test 1 vs Phase 1 (Habit vs Logic)
        # ---------------------------------------------------------
        f.write("## 1. Test 1 vs. Phase 1 (Habit vs. Logic)\n")
        f.write("Percentage shifts (deltas) in evaluation categories between unconstrained generation (Phase 1) and Token-Forcing (Test 1).\n\n")
        
        f.write("| Injection Type | Category | Phase 1 (Natural) | Test 1 (Forced) | Delta |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: |\n")
        
        for inj in injection_types:
            p1_pcts, p1_tot = calc_category_distribution(p1_data, inj)
            t1_pcts, t1_tot = calc_category_distribution(t1_data, inj)
            
            # Print row for each category
            for cat in cats:
                p1_val = p1_pcts.get(cat, 0.0) if p1_tot > 0 else 0.0
                t1_val = t1_pcts.get(cat, 0.0) if t1_tot > 0 else 0.0
                delta = t1_val - p1_val
                sign = "+" if delta > 0 else ""
                f.write(f"| `{inj}` | **{cat}** | {p1_val:.1f}% | {t1_val:.1f}% | {sign}{delta:.1f}% |\n")
        f.write("\n")

        # ---------------------------------------------------------
        # 2. Test 2 vs. Test 1 (Post-Hoc Awareness)
        # ---------------------------------------------------------
        f.write("## 2. Test 2 vs. Test 1 (Post-Hoc Awareness)\n")
        f.write("Evaluating whether the model can retroactively identify and fix a forced error (from Test 1) when explicitly prompted in hindsight.\n\n")
        
        # Calculate Recovery Rate & Double-Down Rate from Test 2 data
        t2_recovered = 0
        t2_doubled_down = 0
        t2_total = len(t2_data)
        
        for row in t2_data:
            # Check for hindsight fix (Recovery) vs stubbornly keeping the error (Double-Down)
            cat = row.get('evaluation_category', '')
            success_flags = row.get('hindsight_successful', False)
            if success_flags or cat in ['Explicit Correction', 'Recovered']:
                t2_recovered += 1
            else:
                t2_doubled_down += 1
                
        recovery_rate = (t2_recovered / t2_total * 100) if t2_total > 0 else 0.0
        double_down_rate = (t2_doubled_down / t2_total * 100) if t2_total > 0 else 0.0
        
        f.write(f"- **Recovery Rate:** {recovery_rate:.1f}% (Post-hoc prompting successfully fixed a forced error)\n")
        f.write(f"- **Double-Down Rate:** {double_down_rate:.1f}% (Model stubbornly defended the forced error)\n\n")

        # ---------------------------------------------------------
        # 3. Test 2 vs. Phase 1 (Forced Post-Hoc vs. Natural Mid-Stream)
        # ---------------------------------------------------------
        f.write("## 3. Test 2 vs. Phase 1 (Forced Post-Hoc vs. Natural Mid-Stream)\n")
        f.write("Comparing the success rate of external post-hoc prompting (Test 2) to the model's natural autonomous correction behavior (Phase 1).\n\n")

        p1_total_ec = sum(1 for r in p1_data if r.get('evaluation_category') == 'Explicit Correction')
        p1_total = len(p1_data)
        p1_ec_rate = (p1_total_ec / p1_total * 100) if p1_total > 0 else 0.0
        
        f.write(f"- **Natural Explicit Correction Rate (Phase 1):** {p1_ec_rate:.1f}%\n")
        f.write(f"- **Forced Post-Hoc Recovery Rate (Test 2):** {recovery_rate:.1f}%\n")
        diff = recovery_rate - p1_ec_rate
        f.write(f"- **Discrepancy (Test 2 - Phase 1):** {diff:+.1f}%\n\n")
        
        if diff > 0:
            f.write("> **Insight:** External prompting yields a **higher** correction rate than the model's natural autonomous behavior.\n\n")
        elif diff < 0:
            f.write("> **Insight:** External prompting yields a **lower** correction rate than natural autonomous behavior.\n\n")
        else:
            f.write("> **Insight:** External prompting and natural autonomous behavior yield **identical** correction rates.\n\n")

        # ---------------------------------------------------------
        # 4. Test 3 vs. Phase 1 (Over-Correction)
        # ---------------------------------------------------------
        f.write("## 4. Test 3 vs. Phase 1 (Over-Correction)\n")
        f.write("Calculating the False-Positive Rate to see if the model incorrectly self-corrects a valid but counter-intuitive premise.\n\n")
        
        anomaly_count = 0
        false_positives = 0
        for row in t3_data:
            if row.get('injection_type') == 'anomaly_weird_true':
                anomaly_count += 1
                cat = row.get('evaluation_category')
                # A false positive is when it "corrects" or rationalizes an anomaly that wasn't actually mathematically/logically wrong
                if cat in ['Explicit Correction', 'Rationalization']:
                    false_positives += 1
                    
        fpr = (false_positives / anomaly_count * 100) if anomaly_count > 0 else 0.0
        
        f.write(f"- **False-Positive Rate (FPR) on Anomalies:** {fpr:.1f}%\n\n")
        
        if fpr > 50.0:
            f.write("> **Insight:** The model frequently panics and over-corrects when encountering odd but factually true statements, suggesting its self-correction is heavily tied to distributional norms rather than strict logical validity.\n")
        else:
            f.write("> **Insight:** The model maintains logical composure and rarely over-corrects unusual but valid statements.\n")

    print(f"Unified analysis report successfully generated and saved to: {output_md}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze Phase 2 Test Suite Results")
    parser.add_argument("--p1", default="data/transcripts/phase1_result_evaluate_adversarial_distilled_fixed.jsonl", help="Phase 1 Baseline JSONL")
    parser.add_argument("--t1", default="data/transcripts/phase2_token_forcing.jsonl", help="Test 1 JSONL")
    parser.add_argument("--t2", default="data/transcripts/phase2_hindsight.jsonl", help="Test 2 JSONL")
    parser.add_argument("--t3", default="data/transcripts/phase2_anomaly.jsonl", help="Test 3 JSONL")
    parser.add_argument("--output", default="data/transcripts/phase2_suite_analysis_report.md", help="Output Markdown report path")
    
    args = parser.parse_args()
    analyze_suite(args.p1, args.t1, args.t2, args.t3, args.output)
