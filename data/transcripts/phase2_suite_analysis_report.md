# Phase 2 Suite Analysis Report

## 1. Test 1 vs. Phase 1 (Habit vs. Logic)
Percentage shifts (deltas) in evaluation categories between unconstrained generation (Phase 1) and Token-Forcing (Test 1).

| Injection Type | Category | Phase 1 (Natural) | Test 1 (Forced) | Delta |
| :--- | :--- | :---: | :---: | :---: |
| `obvious_arithmetic` | **Explicit Correction** | 22.2% | 5.6% | -16.7% |
| `obvious_arithmetic` | **Silent Drift** | 61.1% | 66.7% | +5.6% |
| `obvious_arithmetic` | **Rationalization** | 16.7% | 11.1% | -5.6% |
| `obvious_arithmetic` | **Commits to Error** | 0.0% | 16.7% | +16.7% |
| `plausible_logical` | **Explicit Correction** | 5.6% | 0.0% | -5.6% |
| `plausible_logical` | **Silent Drift** | 88.9% | 88.9% | 0.0% |
| `plausible_logical` | **Rationalization** | 0.0% | 11.1% | +11.1% |
| `plausible_logical` | **Commits to Error** | 5.6% | 0.0% | -5.6% |
| `subtle_numerical` | **Explicit Correction** | 0.0% | 5.6% | +5.6% |
| `subtle_numerical` | **Silent Drift** | 83.3% | 66.7% | -16.7% |
| `subtle_numerical` | **Rationalization** | 16.7% | 11.1% | -5.6% |
| `subtle_numerical` | **Commits to Error** | 0.0% | 16.7% | +16.7% |

## 2. Test 2 vs. Test 1 (Post-Hoc Awareness)
Evaluating whether the model can retroactively identify and fix a forced error (from Test 1) when explicitly prompted in hindsight.

- **Recovery Rate:** 0.0% (Post-hoc prompting successfully fixed a forced error)
- **Double-Down Rate:** 100.0% (Model stubbornly defended the forced error)

## 3. Test 2 vs. Phase 1 (Forced Post-Hoc vs. Natural Mid-Stream)
Comparing the success rate of external post-hoc prompting (Test 2) to the model's natural autonomous correction behavior (Phase 1).

- **Natural Explicit Correction Rate (Phase 1):** 9.3%
- **Forced Post-Hoc Recovery Rate (Test 2):** 0.0%
- **Discrepancy (Test 2 - Phase 1):** -9.3%

> **Insight:** External prompting yields a **lower** correction rate than natural autonomous behavior.

## 4. Test 3 vs. Phase 1 (Over-Correction)
Calculating the False-Positive Rate to see if the model incorrectly self-corrects a valid but counter-intuitive premise.

- **False-Positive Rate (FPR) on Anomalies:** 0.0%

> **Insight:** The model maintains logical composure and rarely over-corrects unusual but valid statements.
