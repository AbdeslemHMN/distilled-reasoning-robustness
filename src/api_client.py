"""
Wrapper around Google Gemini via the official Google GenAI SDK (google-genai).
Used as the RL-native comparison model for adversarial reasoning evaluation.

Exposes the ReasoningModel interface expected by the evaluation harness:
    generate_cot(problem_prompt)      -> full generated text (CoT + answer)
    continue_from(forced_prefix_text) -> continuation generated from that prefix
"""

from __future__ import annotations

import time
from google import genai
from google.genai import types
import config

# --------------------------------------------------------------------------
# Client setup & ThinkingConfig configuration
# --------------------------------------------------------------------------
_gemini_client = None


def get_gemini_client(api_key: str | None = None) -> genai.Client:
    global _gemini_client
    key = api_key or config.GEMINI_API_KEY
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Please set it in your environment or .env file."
        )
    if api_key and api_key != config.GEMINI_API_KEY:
        return genai.Client(api_key=api_key)
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=key)
    return _gemini_client


# --------------------------------------------------------------------------
# API query helper function
# --------------------------------------------------------------------------
def query_gemini_model(
    prompt: str,
    max_tokens: int | None = None,
    model_id: str = config.API_MODEL_ID,
    api_key: str | None = None,
) -> str:
    """Execute inference calling the comparison API using client.models.generate_content."""
    client = get_gemini_client(api_key=api_key)
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=config.THINKING_BUDGET
        ),
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        max_output_tokens=max_tokens or config.MAX_NEW_TOKENS_COT,
    )

    max_retries = 6
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=cfg,
            )
            # Add delay to respect free-tier limits
            time.sleep(3.5)
            return response.text or ""
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                delay = base_delay * (2 ** attempt)
                print(f"[API_CLIENT] Rate limited. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Error querying Gemini API: {e}")
                return ""

    print("[API_CLIENT] Max retries reached.")
    return ""


# --------------------------------------------------------------------------
# Harness Model Adapter
# --------------------------------------------------------------------------
class APIReasoningModel:
    def __init__(
        self,
        model_id: str = config.API_MODEL_ID,
        api_key: str = config.GEMINI_API_KEY,
    ):
        self.model_id = model_id
        self.api_key = api_key
        self.continuation_type = "chat_assistant_prefill"

    def generate_cot(self, formatted_prompt: str) -> str:
        """Generate baseline reasoning trace using Gemini."""
        return query_gemini_model(
            formatted_prompt,
            max_tokens=config.MAX_NEW_TOKENS_COT,
            model_id=self.model_id,
            api_key=self.api_key,
        )

    def continue_from(self, forced_prefix_text: str) -> str:
        """Harmonized injection: pass adversarial injection as structured prompt-level prefill.

        Since the Gemini API does not allow prefilling context directly inside
        an ongoing hidden thinking block, adversarial injections are passed
        as structured prompt-level prefilled contexts.
        """
        prompt, partial_solution = self._split_for_continuation(forced_prefix_text)

        structured_continuation_prompt = (
            f"Problem: {prompt}\n\n"
            f"Given the following partial solution: {partial_solution}, "
            "complete the reasoning and give the final answer."
        )

        return query_gemini_model(
            structured_continuation_prompt,
            max_tokens=config.MAX_NEW_TOKENS_CONTINUATION,
            model_id=self.model_id,
            api_key=self.api_key,
        )


    @staticmethod
    def _split_for_continuation(forced_prefix_text: str) -> tuple[str, str]:
        """Separate the original problem prompt from the partial solution prefix."""
        marker = "\n\nSolution:\n"
        prompt, separator, partial_solution = forced_prefix_text.partition(marker)
        if separator:
            return prompt, partial_solution

        return (
            "Solve the problem.",
            forced_prefix_text,
        )

import os
from openai import OpenAI

# Active Groq model alias mapping
ACTIVE_GROQ_MODEL_ALIAS_MAP = {
    "llama-3.3-70b-versatile": "openai/gpt-oss-120b",
    "qwen-2.5-32b": "qwen/qwen3.6-27b",
    "gpt-oss": "openai/gpt-oss-120b",
    "qwen-27b": "qwen/qwen3.6-27b",
}


def resolve_groq_model(model_id: str) -> str:
    """Map legacy or aliased model strings to active Groq endpoints."""
    return ACTIVE_GROQ_MODEL_ALIAS_MAP.get(model_id, model_id)


def query_groq_model(model_id: str, prompt: str, max_tokens: int | None = None) -> str:
    """Execute inference calling the Groq API using openai client."""
    resolved_model = resolve_groq_model(model_id)
    groq_api_key = (
        os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or config.GROQ_API_KEY
    )
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY (or OPENAI_API_KEY) is missing. Please set it in your environment or .env file."
        )

    client = OpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    # Pause to allow the 1,000 OTPM window to completely reset
    print("\n[GROQ_API] Pacing: pausing 65s to allow 1,000 OTPM rate-limit window to reset...", flush=True)
    time.sleep(65)

    max_retries = 6
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                max_tokens=max_tokens or config.MAX_NEW_TOKENS_COT,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            err_str = str(e)
            if "404" in err_str or "not_found" in err_str.lower() or "model_not_found" in err_str.lower():
                try:
                    available = [m.id for m in client.models.list().data]
                    print(f"\n[GROQ_API ERROR] Model '{resolved_model}' not found (404).")
                    print("Available active models on Groq:\n" + "\n".join(f"  - {m}" for m in available))
                    raise ValueError(
                        f"Groq model '{resolved_model}' returned 404. Available models: {available}"
                    ) from e
                except Exception as list_err:
                    if isinstance(list_err, ValueError):
                        raise list_err
                    print(f"\n[GROQ_API ERROR] Model '{resolved_model}' returned 404, and could not list available models: {list_err}")
                    raise ValueError(f"Groq model '{resolved_model}' returned 404.") from e
            elif "429" in err_str or "rate" in err_str.lower():
                delay = base_delay * (2 ** attempt)
                print(f"[GROQ_API] Rate limited. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"Error querying Groq API: {e}")
                return ""

    print("[GROQ_API] Max retries reached.")
    return ""


class GroqReasoningModel:
    def __init__(self, model_id: str):
        self.model_id = resolve_groq_model(model_id)
        self.continuation_type = "chat_assistant_prefill"

    def generate_cot(self, formatted_prompt: str) -> str:
        """Generate baseline reasoning trace using Groq."""
        return query_groq_model(self.model_id, formatted_prompt, max_tokens=config.MAX_NEW_TOKENS_COT)

    def continue_from(self, forced_prefix_text: str) -> str:
        """Harmonized injection for Groq API."""
        prompt, partial_solution = self._split_for_continuation(forced_prefix_text)

        structured_continuation_prompt = (
            f"Problem: {prompt}\n\n"
            f"Given the following partial solution: {partial_solution}, "
            "complete the reasoning and give the final answer."
        )

        return query_groq_model(
            self.model_id,
            structured_continuation_prompt,
            max_tokens=config.MAX_NEW_TOKENS_CONTINUATION,
        )

    @staticmethod
    def _split_for_continuation(forced_prefix_text: str) -> tuple[str, str]:
        marker = "\n\nSolution:\n"
        prompt, separator, partial_solution = forced_prefix_text.partition(marker)
        if separator:
            return prompt, partial_solution

        return (
            "Solve the problem.",
            forced_prefix_text,
        )

