# GCSE Biology auto-marker

Fine-tune of Mistral-7B-Instruct (QLoRA) to mark short GCSE Biology answers and write feedback. The original work was two Colab notebooks. This repo adds the bits that were missing if I wanted to put it on a CV: data checks, a sklearn baseline you can run on a laptop, and an offline eval.

I am not claiming this is a production marking system. The labelled set is 128 rows.

## What it does

Student answers go in as ChatML (`Question` / `Max Marks` / `Student Answer`). The model is supposed to reply:

```text
Marks: 1/2
Feedback: ...
Tip: ...
```

There are two scoring paths:

1. **TF-IDF + Ridge** in `src/feedbio` — predicts the mark only. Runs locally, covered by tests.
2. **QLoRA on Mistral-7B** in the notebooks — mark + feedback + tip. Needs a Colab GPU and a Hugging Face token.

The baseline is there so the repo is usable without 15GB of VRAM, and so I have something to compare the LLM against.

## Data

`prompt.jsonl` — 128 ChatML examples, 27 unique questions. A few questions are repeated a lot (one appears 20 times). Three rows originally had `Marks: 2/1`; those were clipped to `1/1`.

```text
python scripts/prepare_data.py --data prompt.jsonl --out-dir data
```

That prints dataset stats, writes `data/examples.csv`, and a **question-level** train/test split. A normal row split would leak: train and test would share the same questions.

## Results (baseline, question holdout)

From `python scripts/train_baseline.py` (seed 42, 22 questions train / 5 questions test, 35 test rows):

| Metric | Value |
| --- | --- |
| MAE | 0.83 |
| Exact mark match | 34.3% |
| Within 1 mark | 82.9% |

Full write-up: [reports/baseline_eval.md](reports/baseline_eval.md).

I have not re-run a holdout eval of the QLoRA adapter in this repo. The script for that is `scripts/eval_llm.py` (same metrics, once you dump generations). Trainer loss from the original Colab run is not a marking metric.

## Layout

```text
prompt.jsonl                 labelled ChatML
src/feedbio/                 parse, validate, split, baseline, eval
scripts/                     prepare_data, train_baseline, eval_llm
tests/                       parser, validation, baseline smoke
01_mistral_finetune_pipeline.ipynb
02_inference_demo.ipynb
```

## Local setup

```text
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/prepare_data.py
python scripts/train_baseline.py
```

GPU extras for the notebooks are in `requirements-gpu.txt`. CI runs lint + pytest on the CPU path.

## Colab (QLoRA)

1. Upload `prompt.jsonl` and open `01_mistral_finetune_pipeline.ipynb`.
2. Put a Hugging Face token in Colab secrets as `HF_TOKEN`.
3. The notebook loads Mistral-7B in 4-bit, trains a LoRA adapter for 15 epochs, and copies it to Drive.
4. `02_inference_demo.ipynb` loads that adapter. The system prompt and user template now match training.

Things I would change if I trained again: fewer epochs (128 rows overfit easily), assistant-only loss masking, and the same question-level split as the baseline.

## Limits

- 128 examples, 27 questions, heavy duplication.
- Baseline cannot write feedback; it only predicts a mark, and it is weak on unseen questions.
- QLoRA eval numbers are not in this repo.
- Training still uses full-sequence labels (`labels = input_ids`).
- No mark schemes are stored separately from the teacher feedback in the JSONL.

## CV line

GCSE Biology auto-marker — ChatML dataset checks, sklearn TF-IDF baseline, QLoRA fine-tune of Mistral-7B, offline mark MAE / exact-match eval, pytest + GitHub Actions.
