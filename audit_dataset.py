import sys
import json
import re
import argparse
from pathlib import Path
import sympy

CORRECTION_MARKERS = [
    r"\bwait\b",
    r"\bactually\b",
    r"\bhold on\b",
    r"\bmistake\b",
    r"\berror\b",
    r"\bincorrect\b",
    r"\bwrong\b",
    r"\bre-evaluat",
    r"\blet me check\b",
    r"\bthat can't be right\b",
    r"\bdoesn't make sense\b",
]

def has_explicit_correction_marker(transcript: str) -> bool:
    """Checks for linguistic markers indicating the model spotted an error."""
    text = transcript.lower()
    return any(re.search(pattern, text) for pattern in CORRECTION_MARKERS)

def parse_answer(ans_str: str):
    """
    Parses LaTeX and plain text into SymPy numeric/symbolic expressions.
    """
    if not ans_str:
        return None

    s = ans_str.strip()
    
    # Clean LaTeX inline delimiters and dollar signs
    s = s.replace(r'\(\displaystyle', '').replace(r'\)', '').replace(r'\(', '')
    s = s.replace('$', '').replace(',', '')

    # Clean text formatting blocks with exponents first (e.g. \text{cm}^2, \text{m}^3)
    s = re.sub(r'\\(text|mathrm|mb|textsf)\s*\{[^}]*\}\^\{?[0-9]+\}?', '', s)
    s = re.sub(r'\\(text|mathrm|mb|textsf)\s*\{[^}]*\}', '', s)
    
    # Strip common unit abbreviations with exponents (e.g., cm^2, m^2)
    s = re.sub(r'(?:cm|m|km|mm|in|ft)\s*\^\s*\{?[0-9]+\}?', '', s, flags=re.IGNORECASE)
    
    # Resolve LaTeX fractions
    frac_pattern = re.compile(r'\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}')
    while frac_pattern.search(s):
        s = frac_pattern.sub(r'((\1)/(\2))', s)
        
    s = s.replace(r'\%', '/100').replace('%', '/100')
    
    # Strip remaining letters/words (e.g., 'liters', 'dollars')
    cleaned_math = re.sub(r'[a-zA-Z\s\\]', '', s)

    # Fallback: If stripping letters left an empty string or complex sentence,
    # try extracting the last isolated number (e.g., "receives 4 chocolates" -> "4")
    if not cleaned_math or not any(char.isdigit() for char in cleaned_math):
        numbers = re.findall(r'-?\d+(?:\.\d+)?', ans_str)
        if numbers:
            cleaned_math = numbers[-1]

    try:
        return sympy.sympify(cleaned_math)
    except Exception:
        return None

def extract_boxed(text: str) -> str:
    """Extracts content inside \boxed{...} with fallback to 'Final Answer:' plain text."""
    if not text:
        return None

    # 1. Try LaTeX \boxed{...} first
    matches = list(re.finditer(r'\\boxed\s*\{', text))
    if matches:
        last_match = matches[-1]
        start_idx = last_match.end()
        brace_count = 1
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            
            if brace_count == 0:
                return text[start_idx:i]
                
    # 2. Fallback: Look for plain text "Final Answer", "Final answer:", or "Answer:"
    fallback_match = re.search(r'(?:Final\s*[Aa]nswer|Answer)\s*:?\s*\*?\*?\n*(.+)', text, re.IGNORECASE | re.DOTALL)
    if fallback_match:
        ans_text = fallback_match.group(1).strip()
        ans_text = ans_text.split('\n')[0].strip('. ')
        return ans_text.replace('**', '')

    # 3. Absolute fallback: Just return the last non-empty line of the transcript
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        return lines[-1].replace('**', '').strip('. ')
        
    return None

def check_answer_match(extracted_str: str, ref_ans_str: str) -> bool:
    """Evaluates whether model's extracted answer matches ground truth mathematically or symbolically."""
    if extracted_str is None or ref_ans_str is None:
        return False
        
    extracted_clean = extracted_str.strip().lower()
    ref_clean = ref_ans_str.strip().lower()
    
    # Direct exact string match
    if extracted_clean == ref_clean:
        return True

    # Logic problem heuristics (Yes/No, Valid/Invalid)
    if ref_clean in ["no", "false", "invalid"]:
        if any(w in extracted_clean for w in ["no", "invalid", "not valid", "cannot deduce", "false"]):
            return True
    elif ref_clean in ["yes", "true", "valid"]:
        if any(w in extracted_clean for w in ["yes", "valid", "is true", "true"]):
            return True

    # SymPy symbolic and numeric comparison
    model_val = parse_answer(extracted_str)
    ref_val = parse_answer(ref_ans_str)
    
    if model_val is not None and ref_val is not None:
        try:
            # 1. Exact symbolic match
            diff = sympy.simplify(model_val - ref_val)
            if diff == 0:
                return True
            # 2. Float tolerance match
            if abs(float(model_val.evalf()) - float(ref_val.evalf())) < 1e-5:
                return True
        except Exception:
            pass

    # Numeric sequence fallback
    extracted_nums = re.findall(r'-?\d+(?:\.\d+)?', extracted_str)
    ref_nums = re.findall(r'-?\d+(?:\.\d+)?', ref_ans_str)
    if extracted_nums and ref_nums:
        try:
            ref_float = float(ref_nums[-1])
            if any(abs(float(num) - ref_float) < 1e-5 for num in [extracted_nums[-1]] + extracted_nums[:-1]):
                return True
        except Exception:
            pass

    return False

def determine_corrected_category(record: dict, is_match: bool) -> str:
    """Determines the appropriate category based on Phase 1 taxonomy heuristics."""
    continuation = record.get("forced_continuation", "")
    injected_str = record.get("injected_sentence", "")

    if is_match:
        if has_explicit_correction_marker(continuation):
            return "Explicit Correction"
        return "Silent Drift"
    else:
        injected_numbers = re.findall(r"-?\d+(?:\.\d+)?", injected_str)
        cont_numbers = re.findall(r"-?\d+(?:\.\d+)?", continuation)
        uses_injected_num = any(num in cont_numbers for num in injected_numbers if float(num) > 1)

        if uses_injected_num and len(continuation) > 250:
            return "Rationalization"
        return "Commits to Error"

def main():
    parser = argparse.ArgumentParser(description="Audit and optionally fix Phase 1 JSONL output transcripts.")
    parser.add_argument("file", nargs="?", default="data/transcripts/phase1_result_evaluate_api_openai_gpt_oss_120b.jsonl", help="JSONL file to audit")
    parser.add_argument("--fix", action="store_true", help="Auto-correct mislabeled records using intelligent heuristics and save to a clean JSONL file.")
    parser.add_argument("--output", type=str, default=None, help="Output path for the fixed JSONL file (defaults to <input_stem>_fixed.jsonl)")
    args = parser.parse_args()
    
    input_path = Path(args.file)
    if not input_path.exists():
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)
        
    records = []
    mislabeled_false_positives = []
    mislabeled_uncaught_fails = []
    unparseable = []
    fixes_applied = []
    total_records = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total_records += 1
            
            prob_id = record.get("problem_id", f"Row {i}")
            forced_cont = record.get("forced_continuation", "")
            ref_ans_str = record.get("problem_reference_answer", "")
            current_cat = record.get("evaluation_category", "")
            
            extracted_str = extract_boxed(forced_cont)
            if extracted_str is None:
                unparseable.append((i, prob_id))
                records.append(record)
                continue
                
            is_match = check_answer_match(extracted_str, ref_ans_str)
            
            # Check category mapping
            is_labeled_incorrect = current_cat in ["Commits to Error", "Rationalization"]
            is_labeled_correct = current_cat in ["Explicit Correction", "Silent Drift"]
            
            if is_match and is_labeled_incorrect:
                new_cat = determine_corrected_category(record, is_match=True)
                mislabeled_false_positives.append((i, prob_id, extracted_str, ref_ans_str, current_cat, new_cat))
                if args.fix:
                    record["evaluation_category"] = new_cat
                    fixes_applied.append((i, prob_id, current_cat, new_cat))
            elif not is_match and is_labeled_correct:
                # Uncaught Fails: Flagged for audit report but NEVER auto-modified in the output JSONL
                new_cat = determine_corrected_category(record, is_match=False)
                mislabeled_uncaught_fails.append((i, prob_id, extracted_str, ref_ans_str, current_cat, new_cat))
                    
            records.append(record)

    # Output report
    print(f"--- Audit Report for: {input_path} ---")
    print(f"Total Records Audited: {total_records}")
    print(f"Total Unparseable (No \\boxed{{...}} found): {len(unparseable)}")
    print(f"Total Mislabeled (False Positive - Labeled Error but math is correct): {len(mislabeled_false_positives)}")
    print(f"Total Mislabeled (Uncaught Fail - Labeled Correct but math is wrong): {len(mislabeled_uncaught_fails)}\n")
    
    if unparseable:
        print("--- Unparseable Output ---")
        for idx, pid in unparseable:
            print(f"  Index {idx} | Problem ID: {pid}")
            
    if mislabeled_false_positives:
        print("\n--- Mislabeled: False Positives (Math correct, categorized as Error) ---")
        for idx, pid, ex, ref, cat, new_cat in mislabeled_false_positives:
            fix_msg = f" -> Fixed to '{new_cat}'" if args.fix else f" [Suggest: '{new_cat}']"
            print(f"  Index {idx} | Problem ID: {pid} | Labeled: '{cat}'{fix_msg} | Model Ans: '{ex}' | Ref Ans: '{ref}'")

    if mislabeled_uncaught_fails:
        print("\n--- Mislabeled: Uncaught Fails (Math incorrect, categorized as Correct) ---")
        for idx, pid, ex, ref, cat, new_cat in mislabeled_uncaught_fails:
            print(f"  Index {idx} | Problem ID: {pid} | Labeled: '{cat}' [Kept As-Is: Manual Review] | Model Ans: '{ex}' | Ref Ans: '{ref}'")

    # If --fix was specified, write fixed records
    if args.fix:
        if args.output:
            out_file = Path(args.output)
        else:
            out_file = input_path.with_name(f"{input_path.stem}_fixed.jsonl")
            
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w', encoding='utf-8') as f_out:
            for rec in records:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
        print("\n" + "=" * 65)
        print(f"[FIX COMPLETE] Corrected {len(fixes_applied)} False Positive records.")
        if mislabeled_uncaught_fails:
            print(f"[PRESERVED] {len(mislabeled_uncaught_fails)} Uncaught Fail records left unmodified for manual review.")
        print(f"[OUTPUT FILE] Saved clean dataset to: {out_file}")
        print("=" * 65)

if __name__ == "__main__":
    main()
