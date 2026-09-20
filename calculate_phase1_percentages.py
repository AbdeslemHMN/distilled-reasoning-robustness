#!/usr/bin/env python3
"""
calculate_phase1_percentages.py

Parses the Phase 1 comparison markdown table (e.g., phase1_comparison_table.md)
and computes counts and percentage distributions for each behavioral category:
- Overall percentage per model (Distilled vs API/RL Target)
- Breakdown by Injection Type (obvious_arithmetic, plausible_logical, subtle_numerical)
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_comparison_table(file_path: Path):
    if not file_path.exists():
        print(f"[ERROR] Table file not found: {file_path}")
        sys.exit(1)

    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, separators, and non-table lines
            if not line.startswith("|") or ":---" in line:
                continue

            # Split cells
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) < 4:
                continue

            # Skip header line
            if cells[0].lower() == "problem" and "injection" in cells[1].lower():
                continue

            problem_id = cells[0]
            inj_type = cells[1].strip("` ")
            distilled_cat = cells[2].strip()
            api_cat = cells[3].strip()

            rows.append({
                "problem": problem_id,
                "injection_type": inj_type,
                "distilled": distilled_cat,
                "api": api_cat,
            })

    return rows


def compute_statistics(rows):
    total = len(rows)
    if total == 0:
        return {}, {}, {}, 0

    dist_overall = Counter(r["distilled"] for r in rows)
    api_overall = Counter(r["api"] for r in rows)

    by_inj_dist = defaultdict(Counter)
    by_inj_api = defaultdict(Counter)
    inj_totals = Counter()

    for r in rows:
        inj = r["injection_type"]
        inj_totals[inj] += 1
        by_inj_dist[inj][r["distilled"]] += 1
        by_inj_api[inj][r["api"]] += 1

    return {
        "total": total,
        "dist_overall": dist_overall,
        "api_overall": api_overall,
        "inj_totals": inj_totals,
        "by_inj_dist": by_inj_dist,
        "by_inj_api": by_inj_api,
    }


def print_summary(stats):
    total = stats["total"]
    if total == 0:
        print("[WARNING] No valid data rows found.")
        return

    print("=" * 72)
    print(f"  PHASE 1 COMPARISON TABLE ANALYSIS (Total Instances: {total})")
    print("=" * 72)

    # 1. Overall Model Behavior
    print("\n--- 1. OVERALL BEHAVIOR DISTRIBUTION ---")
    header = f"{'Behavioral Category':<25} | {'Distilled Count (%)':<20} | {'API Target Count (%)':<20}"
    print(header)
    print("-" * len(header))

    all_categories = sorted(list(set(
        list(stats["dist_overall"].keys()) + list(stats["api_overall"].keys())
    )))

    for cat in all_categories:
        d_cnt = stats["dist_overall"][cat]
        d_pct = (d_cnt / total) * 100.0
        a_cnt = stats["api_overall"][cat]
        a_pct = (a_cnt / total) * 100.0
        print(f"{cat:<25} | {f'{d_cnt} ({d_pct:5.1f}%)':<20} | {f'{a_cnt} ({a_pct:5.1f}%)':<20}")

    # 2. Breakdown per Injection Type
    print("\n--- 2. BREAKDOWN BY INJECTION TYPE ---")
    for inj_type in sorted(stats["inj_totals"].keys()):
        inj_tot = stats["inj_totals"][inj_type]
        print(f"\n[Injection Type: '{inj_type}'] (N = {inj_tot}):")
        sub_header = f"  {'Category':<23} | {'Distilled (%)':<18} | {'API Target (%)':<18}"
        print(sub_header)
        print("  " + "-" * (len(sub_header) - 2))

        inj_cats = sorted(list(set(
            list(stats["by_inj_dist"][inj_type].keys()) +
            list(stats["by_inj_api"][inj_type].keys())
        )))

        for cat in inj_cats:
            d_cnt = stats["by_inj_dist"][inj_type][cat]
            d_pct = (d_cnt / inj_tot) * 100.0 if inj_tot else 0.0
            a_cnt = stats["by_inj_api"][inj_type][cat]
            a_pct = (a_cnt / inj_tot) * 100.0 if inj_tot else 0.0
            print(f"  {cat:<23} | {f'{d_cnt} ({d_pct:5.1f}%)':<18} | {f'{a_cnt} ({a_pct:5.1f}%)':<18}")

    print("\n" + "=" * 72)


def generate_markdown_report(stats) -> str:
    total = stats["total"]
    lines = [
        "## Phase 1 Quantitative Summary: Behavioral Percentages",
        f"\n**Total evaluated instances:** {total}\n",
        "### 1. Overall Category Distribution",
        "| Behavioral Category | Distilled Count | Distilled % | API Target Count | API Target % |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    all_categories = sorted(list(set(
        list(stats["dist_overall"].keys()) + list(stats["api_overall"].keys())
    )))

    for cat in all_categories:
        d_cnt = stats["dist_overall"][cat]
        d_pct = (d_cnt / total) * 100.0 if total else 0.0
        a_cnt = stats["api_overall"][cat]
        a_pct = (a_cnt / total) * 100.0 if total else 0.0
        lines.append(f"| {cat} | {d_cnt} | {d_pct:.1f}% | {a_cnt} | {a_pct:.1f}% |")

    lines.append("\n### 2. Distribution by Injection Type\n")
    for inj_type in sorted(stats["inj_totals"].keys()):
        inj_tot = stats["inj_totals"][inj_type]
        lines.append(f"#### Injection: `{inj_type}` (N = {inj_tot})")
        lines.append("| Category | Distilled Count | Distilled % | API Target Count | API Target % |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")

        inj_cats = sorted(list(set(
            list(stats["by_inj_dist"][inj_type].keys()) +
            list(stats["by_inj_api"][inj_type].keys())
        )))
        for cat in inj_cats:
            d_cnt = stats["by_inj_dist"][inj_type][cat]
            d_pct = (d_cnt / inj_tot) * 100.0 if inj_tot else 0.0
            a_cnt = stats["by_inj_api"][inj_type][cat]
            a_pct = (a_cnt / inj_tot) * 100.0 if inj_tot else 0.0
            lines.append(f"| {cat} | {d_cnt} | {d_pct:.1f}% | {a_cnt} | {a_pct:.1f}% |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate counts and percentages from Phase 1 comparison table."
    )
    parser.add_argument(
        "--table",
        default="data/transcripts/phase1_comparison_table.md",
        help="Path to Phase 1 comparison markdown table (default: data/transcripts/phase1_comparison_table.md)",
    )
    parser.add_argument(
        "--markdown_output",
        default=None,
        help="Optional path to save markdown formatted summary report.",
    )
    args = parser.parse_args()

    table_path = Path(args.table)
    rows = parse_comparison_table(table_path)
    stats = compute_statistics(rows)

    print_summary(stats)

    if args.markdown_output:
        md_content = generate_markdown_report(stats)
        out_path = Path(args.markdown_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_content + "\n")
        print(f"[INFO] Markdown summary saved to: {args.markdown_output}")


if __name__ == "__main__":
    main()

