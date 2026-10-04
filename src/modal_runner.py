"""
Modal Cloud GPU runner and adapter for the distilled reasoning model.

Provides:
  - DistilledModelRunner: Remote Modal class running on T4 GPU.
  - ModalReasoningModel: Adapter conforming to src.harness.ReasoningModel protocol.
"""

from __future__ import annotations

import modal
import config
from typing import List, Optional

inference_image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers>=4.44.0",
        "torch>=2.2.0",
        "accelerate>=0.33.0",
        "python-dotenv>=1.0.0",
    )
    .add_local_python_source("config")
    .add_local_python_source("phase2_utils")
)

app = modal.App("cot-adversarial-harness")

@app.cls(gpu="T4", image=inference_image, timeout=600)
class DistilledModelRunner:
    @modal.enter()
    def setup(self):
        """Loads tokenizer and model once into GPU memory when container boots."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import config

        print(f"[Modal T4] Initializing container with {config.LOCAL_MODEL_ID} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.LOCAL_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print("[Modal T4] Model loaded on GPU and ready.")

    @modal.method()
    def generate(
        self,
        prompt_text: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        suppress_tokens: List[int] | None = None,
    ) -> str:
        """Raw text-in, text-out generation using GPU."""
        import torch
        import config
        from transformers import LogitsProcessorList
        from phase2_utils import TokenSuppressionLogitsProcessor

        max_tokens = max_new_tokens or config.MAX_NEW_TOKENS_COT
        temp = temperature if temperature is not None else config.TEMPERATURE
        p = top_p if top_p is not None else config.TOP_P

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to("cuda")
        
        logits_processor = LogitsProcessorList()
        if suppress_tokens:
            logits_processor.append(TokenSuppressionLogitsProcessor(suppressed_token_ids=suppress_tokens))

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temp,
                top_p=p,
                pad_token_id=self.tokenizer.eos_token_id,
                logits_processor=logits_processor,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    @modal.method()
    def get_suppressed_token_ids(self, words: List[str]) -> List[int]:
        """Expose tokenizer to get token IDs remotely without loading model locally."""
        from phase2_utils import get_suppressed_token_ids
        return get_suppressed_token_ids(self.tokenizer, words)


class ModalReasoningModel:
    """Wraps the remote Modal GPU runner so it fits directly into src/harness.py."""

    def __init__(self, runner: DistilledModelRunner | None = None):
        self.runner = runner if runner is not None else DistilledModelRunner()
        self.model_id = f"{config.LOCAL_MODEL_ID} (Modal T4 GPU)"
        self.continuation_type = "true_token_prefill"

    def generate_cot(self, formatted_prompt: str) -> str:
        """Prompts the cloud GPU to generate the initial CoT."""
        return self.runner.generate.remote(
            formatted_prompt,
            max_new_tokens=config.MAX_NEW_TOKENS_COT,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
        )

    def continue_from(self, forced_prefix_text: str, suppress_tokens: List[int] | None = None) -> str:
        """Sends the forced injected prefix to the cloud GPU for token continuation."""
        return self.runner.generate.remote(
            forced_prefix_text,
            max_new_tokens=config.MAX_NEW_TOKENS_CONTINUATION,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            suppress_tokens=suppress_tokens,
        )

    def generate_response(self, prompt: str) -> str:
        """Standard text prompting for single-turn Q&A without CoT specific configs."""
        return self.runner.generate.remote(
            prompt,
            max_new_tokens=256, # Default short answer
            temperature=0.0, # Deterministic answers
            top_p=None,
        )
