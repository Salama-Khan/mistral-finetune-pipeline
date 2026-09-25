from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from feedbio.baseline import fit_baseline, predict_marks
from feedbio.data import load_examples, split_by_question
from feedbio.evaluate import evaluate_marks, format_report
from feedbio.features import dataset_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TF-IDF + Ridge mark baseline")
    parser.add_argument("--data", default="prompt.jsonl")
    parser.add_argument("--model-out", default="models/baseline.joblib")
    parser.add_argument("--report-out", default="reports/baseline_eval.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    examples = [ex for ex in load_examples(args.data) if ex.is_valid]
    train, test = split_by_question(examples, test_size=args.test_size, seed=args.seed)
    pipe = fit_baseline(train)
    preds = predict_marks(pipe, test)
    result = evaluate_marks(test, preds)

    split_note = (
        f"Question-level holdout, seed={args.seed}, test_size={args.test_size}. "
        f"{len(train)} train rows ({len({e.question for e in train})} questions), "
        f"{len(test)} test rows ({len({e.question for e in test})} questions)."
    )
    report = format_report(
        "Baseline eval (TF-IDF + Ridge)",
        dataset_stats(examples),
        split_note,
        result,
    )

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.model_out)
    Path(args.report_out).write_text(report)

    print(report)
    print(f"wrote {args.model_out}")
    print(f"wrote {args.report_out}")


if __name__ == "__main__":
    main()
