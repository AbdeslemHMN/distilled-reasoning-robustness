"""
Wrapper around a locally-loaded HF reasoning model (e.g. the DeepSeek-R1 distill).

Exposes exactly two methods so it's interchangeable with APIReasoningModel
in the harness:

    generate_cot(problem_prompt)      -> full generated text (CoT + answer)
    continue_from(forced_prefix_text) -> continuation generated from that prefix

Everything else (sentence splitting, injection, logging) lives elsewhere so
this file only has to know how to talk to the model.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


class LocalReasoningModel:
    def __init__(
        self,
        model_id: str = config.LOCAL_MODEL_ID,
        device: str = config.DEVICE,
        dtype=config.DTYPE,
    ):
        self.model_id = model_id
        self.device = device
        self.continuation_type = "true_token_prefill"

        print(f"[LocalReasoningModel] Loading tokenizer for {model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        print(f"[LocalReasoningModel] Loading model {model_id} on {device} ({dtype}) ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device if device == "cuda" else None,
        )
        if device != "cuda":
            self.model.to(device)
        self.model.eval()
        print("[LocalReasoningModel] Ready.")

    # ------------------------------------------------------------------
    # Core generation methods
    # ------------------------------------------------------------------
    def _generate(self, prompt_text: str, max_new_tokens: int, temperature: float = config.TEMPERATURE, suppress_tokens: list[int] | None = None) -> str:
        """Raw text-in, text-out generation. No chat template gymnastics —
        we want full control over the literal token stream so we can inject
        text mid-generation later."""
        from transformers import LogitsProcessorList
        from phase2_utils import TokenSuppressionLogitsProcessor
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        
        logits_processor = LogitsProcessorList()
        if suppress_tokens:
            logits_processor.append(TokenSuppressionLogitsProcessor(suppressed_token_ids=suppress_tokens))
            
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=config.TOP_P if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                logits_processor=logits_processor,
            )
        # Only decode the newly generated continuation, not the echoed prompt.
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_cot(self, formatted_prompt: str) -> str:
        """Generate a full chain-of-thought + answer for a fresh problem.
        Returns the raw generated text (prompt is NOT included)."""
        return self._generate(formatted_prompt, config.MAX_NEW_TOKENS_COT)

    def continue_from(self, forced_prefix_text: str, suppress_tokens: list[int] | None = None) -> str:
        return self._generate(forced_prefix_text, config.MAX_NEW_TOKENS_CONTINUATION, suppress_tokens=suppress_tokens)
        
    def generate_response(self, prompt: str) -> str:
        return self._generate(prompt, 128, temperature=0.0)
        
    def get_suppressed_token_ids(self, words: list[str]) -> list[int]:
        from phase2_utils import get_suppressed_token_ids
        return get_suppressed_token_ids(self.tokenizer, words)
