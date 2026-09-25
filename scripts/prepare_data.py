from __future__ import annotations

import argparse
import json
from pathlib import Path

from feedbio.data import load_examples, split_by_question
from feedbio.features import dataset_stats, to_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ChatML JSONL and write splits")
    parser.add_argument("--data", default="prompt.jsonl")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    examples = load_examples(args.data)
    stats = dataset_stats(examples)
    print(json.dumps(stats, indent=2))

    invalid = [ex for ex in examples if not ex.is_valid]
    if invalid:
        print(f"\n{len(invalid)} invalid rows:")
        for ex in invalid:
            print(f"- {ex.question!r}: {ex.issues}")

    valid = [ex for ex in examples if ex.is_valid]
    train, test = split_by_question(valid, test_size=args.test_size, seed=args.seed)
    print(f"\nsplit: {len(train)} train / {len(test)} test rows")
    print(f"questions: {len({e.question for e in train})} train / {len({e.question for e in test})} test")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    to_dataframe(valid).to_csv(out_dir / "examples.csv", index=False)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "test.jsonl", test)


def _write_jsonl(path: Path, examples) -> None:
    with path.open("w") as handle:
        for ex in examples:
            handle.write(json.dumps({"messages": ex.messages}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
