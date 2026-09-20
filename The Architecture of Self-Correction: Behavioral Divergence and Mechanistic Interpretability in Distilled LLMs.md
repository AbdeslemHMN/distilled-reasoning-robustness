# Phase 1 Behavioral Evaluation: The Impact of Distillation on LLM Self-Correction

## Abstract
This report details the Phase 1 behavioral baseline of an experiment investigating how Large Language Models (LLMs) recover from forced reasoning errors. By injecting false premises into the generation context, we compared a large RL-native teacher model (API Target Model) against a smaller distilled model (1.5B Distilled Model). The results demonstrate a significant behavioral divergence: the RL-native model possesses a 100% robust "Silent Drift" recovery mechanism, whereas the distilled model exhibits higher volatility, substituting silent recovery with both surface-level "Explicit Corrections" and hallucinatory "Rationalizations." This confirms that distillation alters error-handling behavior, paving the way for Phase 2 mechanistic interpretability.

## Introduction
Self-correction in LLMs is a critical capability for reliable reasoning, yet the mechanisms underlying it—and how model distillation impacts it—remain underexplored. This phase establishes a qualitative baseline by observing how models respond to forced prompt injections (errors forcefully inserted into their chain-of-thought). Before analyzing internal layer activations (Phase 2), we must first categorize external behavior to understand if and how self-correction manifests across different error types and model architectures.

## Research Questions
1. Does self-correction happen reliably enough to study mechanistically?
2. Does the model's error-recovery behavior differ by the type of forced injection (obvious arithmetic, subtle numerical, plausible logical)?
3. Does the error-recovery behavior differ between the small distilled model and the large RL-native teacher?

## Hypotheses
*   **H1 (Mechanistic - To be tested in Phase 2):** Models utilize specific internal circuits (e.g., attention heads or residual stream shifts) to detect and override forced errors in context.
*   **H2 (Behavioral - Tested in Phase 1):** The distillation process alters the model's self-correction capabilities, changing how it reacts to and recovers from injected errors compared to its RL-native teacher.

## Methodology: What We Did
We evaluated a dataset of 18 logic and arithmetic problems[cite: 3]. For each problem, we applied three distinct prompt injection types:
*   `obvious_arithmetic` (e.g., stating 2+2=5)
*   `subtle_numerical` (e.g., slightly altering a subtotal)
*   `plausible_logical` (e.g., introducing a logically sounding but false premise)

These 54 variations were run through both the RL-Native API Target Model and the 1.5B Distilled Model[cite: 3]. Transcripts were manually categorized into one of four recovery states:
1.  **Explicit Correction:** Model flags the error and corrects it.
2.  **Silent Drift:** Model ignores the error and quietly outputs the correct final answer.
3.  **Rationalization:** Model gets the final answer wrong by inventing a logical excuse to justify the injected error.
4.  **Commits to Error:** Model gets the final answer wrong by passively accepting the injected error.

## Results: What We Got
The 54 trials per model yielded the following distribution[cite: 3]:

| Behavior Category | API Target Model (RL-Native) | Distilled Model (1.5B) |
| :--- | :--- | :--- |
| **Silent Drift** | 54 (100.0%) | 42 (77.8%) |
| **Explicit Correction** | 0 (0.0%) | 5 (9.3%) |
| **Rationalization** | 0 (0.0%) | 6 (11.1%) |
| **Commits to Error** | 0 (0.0%) | 1 (1.9%) |

## Conclusions & Answers to Core Questions

**Does self-correction happen reliably enough to study?**
Yes. The RL-native model demonstrated a 100% recovery rate via Silent Drift across all 54 trials[cite: 3]. The distilled model successfully recovered in 87.1% of cases (Silent Drift + Explicit Correction)[cite: 3]. This proves the phenomenon is highly reliable and warrants mechanistic investigation.

**Does it differ by injection type?**
Yes, particularly in the distilled model. The distilled model only produced Explicit Corrections primarily in response to `obvious_arithmetic` errors (e.g., problems `arith_03`, `arith_04`, `arith_11`, `logic_02`)[cite: 3]. Conversely, it fell into Rationalization when faced with `subtle_numerical` errors in arithmetic (e.g., `arith_10`, `arith_12`) or when obvious arithmetic was injected into logic problems (e.g., `logic_01`, `logic_05`, `logic_06`)[cite: 3]. 

**Does it differ between the distilled model and the RL-native teacher?**
Significantly. The RL-native model never verbally acknowledged an error, utilizing Silent Drift 100% of the time[cite: 3]. The distilled model lost this absolute resilience. It became more "vocal" by explicitly correcting errors (9.3% of the time), but also more fragile, failing to recover in 13% of cases (Rationalization + Commits to Error)[cite: 3]. This is a citable H2 result confirming distillation fundamentally alters error-handling behavior.

**Interpretation of Results (Meaning)**
The results indicate that the distilled model possesses a more surface-level "catch the error" behavior. Because it Explicitly Corrects obvious errors but Rationalizes subtle ones[cite: 3], its self-correction acts partially as pattern-matching ("this sentence looks like a math mistake") rather than the deep, ingrained reasoning verification exhibited by the teacher model's universal Silent Drift.

**Checkpoint / Kill Criterion Status**
**Passed.** The categorization was clean and unambiguous. The models responded to the injections reliably well before the 4-hour mark. We do not need to narrow the injection design; we are clear to proceed to Phase 2 to uncover the neural mechanisms behind Silent Drift and Explicit Correction.