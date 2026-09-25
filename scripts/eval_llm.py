from __future__ import annotations

import argparse
import json
from pathlib import Path

from feedbio.data import example_from_record
from feedbio.evaluate import evaluate_generations, format_report
from feedbio.features import dataset_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score dumped LLM generations against a gold JSONL holdout"
    )
    parser.add_argument("--gold", required=True, help="ChatML JSONL used as labels")
    parser.add_argument(
        "--preds",
        required=True,
        help="JSONL with a 'text' field per row, same order as --gold",
    )
    parser.add_argument("--report-out", default="reports/llm_eval.md")
    args = parser.parse_args()

    gold = [example_from_record(json.loads(line)) for line in Path(args.gold).read_text().splitlines() if line.strip()]
    preds = [json.loads(line)["text"] for line in Path(args.preds).read_text().splitlines() if line.strip()]
    if len(gold) != len(preds):
        raise SystemExit(f"gold has {len(gold)} rows, preds has {len(preds)}")

    result = evaluate_generations(gold, preds)
    report = format_report(
        "LLM eval",
        dataset_stats(gold),
        f"Scored {len(gold)} dumped generations from {args.preds}.",
        result,
    )
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
