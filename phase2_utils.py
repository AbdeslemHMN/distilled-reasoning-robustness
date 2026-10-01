"""
Phase 2 Utilities & Shared Toolbox.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import LogitsProcessor
import torch

def load_problems(json_path: str = "data/problems/sample_problems.json") -> List[Dict[str, Any]]:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Problem file {json_path} not found.")
    with open(path, "r") as f:
        return json.load(f)

def load_transcripts(jsonl_path: str, filter_categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    path = Path(jsonl_path)
    if not path.exists():
        logging.warning(f"Transcript file {jsonl_path} not found.")
        return []
    
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if filter_categories:
                if record.get("evaluation_category") in filter_categories:
                    records.append(record)
            else:
                records.append(record)
    return records

def save_transcript(output_path: str, record: Dict[str, Any]):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# ==============================================================================
# Custom LogitsProcessor for fine-grained token suppression
# ==============================================================================
class TokenSuppressionLogitsProcessor(LogitsProcessor):
    """
    Penalizes specific token IDs by setting their logits to -infinity.
    Allows for fine-grained control over which tokens are suppressed during generation.
    """
    def __init__(self, suppressed_token_ids: List[int], penalty: float = -float("inf")):
        self.suppressed_token_ids = suppressed_token_ids
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for token_id in self.suppressed_token_ids:
            scores[:, token_id] = self.penalty
        return scores

def get_suppressed_token_ids(tokenizer, words: List[str]) -> List[int]:
    """Helper to find all possible token variations of the given words."""
    suppressed_ids = set()
    for word in words:
        # Tokenize with and without leading space to catch all variations
        variations = [word, f" {word}", word.lower(), f" {word.lower()}"]
        for var in variations:
            # Add special tokens = False to avoid BOS/EOS
            ids = tokenizer.encode(var, add_special_tokens=False)
            if ids:
                # We suppress the FIRST token of the sequence
                suppressed_ids.add(ids[0])
    return list(suppressed_ids)
