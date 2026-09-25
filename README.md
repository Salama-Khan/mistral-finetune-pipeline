# GCSE Biology auto-marker

QLoRA fine-tune of Mistral-7B-Instruct that marks short GCSE Biology answers and writes feedback. The same repo has a sklearn baseline so you can run the mark-prediction path on a laptop.

Student answers are ChatML:

```text
Question: ...
Max Marks: 2
Student Answer: ...
```

The model replies:

```text
Marks: 1/2
Feedback: ...
Tip: ...
```

## Scoring paths

1. **TF-IDF + Ridge** (`src/feedbio`) — predicts the mark only. Runs locally; covered by tests.
2. **QLoRA on Mistral-7B** (the notebooks) — mark, feedback, and tip. Needs a Colab GPU and a Hugging Face token.

The baseline exists so the CPU path is reproducible, and so there is a comparison for the LLM.

## Data

`prompt.jsonl` — 128 labelled examples, 27 unique questions. Some questions are repeated (one appears 20 times).

```text
python scripts/prepare_data.py --data prompt.jsonl --out-dir data
```

That writes `data/examples.csv` and a question-level train/test split. A random row split would put the same questions in both sides.

## Results

TF-IDF + Ridge, question holdout, seed 42 (22 questions train / 5 test, 35 test rows):

| Metric | Value |
| --- | --- |
| MAE | 0.83 |
| Exact mark match | 34.3% |
| Within 1 mark | 82.9% |

Details: [reports/baseline_eval.md](reports/baseline_eval.md).

`scripts/eval_llm.py` scores dumped Mistral generations with the same metrics.

## Setup

```text
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/prepare_data.py
python scripts/train_baseline.py
```

GPU extras for the notebooks are in `requirements-gpu.txt`. CI runs lint and pytest on the CPU path.

## Colab

1. Upload `prompt.jsonl` and open `01_mistral_finetune_pipeline.ipynb`.
2. Add a Hugging Face token in Colab secrets as `HF_TOKEN`.
3. The notebook loads Mistral-7B in 4-bit, trains a LoRA adapter, and copies it to Drive.
4. `02_inference_demo.ipynb` loads that adapter. Prompt format matches training.

If I trained the 7B model again I would use this question-level holdout and mask the prompt tokens in the loss.
