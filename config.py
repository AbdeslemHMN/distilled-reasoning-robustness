"""
Central configuration for the CoT self-correction harness.

"""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
PROBLEMS_PATH = DATA_DIR / "problems" / "sample_problems.json"
SMOKE_TEST_PROBLEMS_PATH = DATA_DIR / "problems" / "smoke_test_problems.json"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Local model (the distilled reasoning model)
# --------------------------------------------------------------------------
LOCAL_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# --------------------------------------------------------------------------
# API model (the RL-native comparison model, via Gemini or Groq)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Gemini model string
API_MODEL_ID = os.environ.get("API_MODEL_ID", "gemini-2.0-flash")
THINKING_BUDGET = int(os.environ.get("THINKING_BUDGET", "2048"))

# --------------------------------------------------------------------------
# Sampling parameters
# --------------------------------------------------------------------------
MAX_NEW_TOKENS_COT = 1024       # budget for the initial full CoT generation
# Reasoning-capable hosted models may spend several hundred tokens thinking
# before emitting visible text, so leave enough room for the continuation too.
MAX_NEW_TOKENS_CONTINUATION = 1024
TEMPERATURE = 0.7
TOP_P = 0.95
SEED = 0

# --------------------------------------------------------------------------
# Injection settings
# --------------------------------------------------------------------------
# Which sentence (by fractional position in the CoT, 0=start, 1=end) to target
# by default when the harness isn't given an explicit index.
DEFAULT_INJECTION_FRACTION = 0.5
