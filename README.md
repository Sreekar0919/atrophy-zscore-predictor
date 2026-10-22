# Atrophy Z-Score Predictor

Can an LLM read a neurology clinical note and predict *where* the brain is atrophied? This project
asks a model to output a ranked list of brain regions with atrophy z-scores. It then compares those
predictions region by region against MRI-derived z-scores. Built during a summer 2025 internship at
UCSF, with a focus on semantic-variant primary progressive aphasia.

## How it works

1. **Diagnosis step.** A reasoning model (`o4-mini`) reads the note and proposes a likely diagnosis.
2. **Structured prediction.** `gpt-4o-mini` returns JSON that follows a fixed schema: rank, region
   (Destrieux atlas names), hemisphere, z-score. Token **logprobs** give a confidence for each value.
3. **Matching.** Predicted region names are mapped to the atlas, with fuzzy matching and manual fixes.
4. **Agreement.** Correlation and regression between predicted and MRI z-scores, plus Bland–Altman
   plots (`analysis/descriptions/compare_combined.py`).

## Layout

```
analysis/
  pipeline.py                 the experiment end to end
  llm_analysis.py             prompt variants
  descriptions/               note generator, JSON schema, comparison stats
  synthetic_data/             synthetic atrophy profile, note and predictions (no patient data)
  test_code/, future_work/    earlier comparison scripts and next steps
package/                      atrophy_predictor: the same idea as an installable package
                              with a CLI (Anthropic API backend)
```

## Run

```bash
pip install -r analysis/requirements.txt
export OPENAI_API_KEY=...
cd analysis && python pipeline.py

pip install -e package          # optional package version
```

All data in this repo is synthetic. Real patient notes and scans stayed inside UCSF.
