from __future__ import annotations

from collections import Counter

import pandas as pd

from .data import Example


def to_dataframe(examples: list[Example]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question": ex.question,
            "max_marks": ex.max_marks,
            "student_answer": ex.student_answer,
            "awarded": ex.awarded,
            "n_issues": len(ex.issues),
            "issues": "; ".join(ex.issues),
        }
        for ex in examples
    )


def dataset_stats(examples: list[Example]) -> dict:
    questions = [ex.question for ex in examples if ex.question]
    awarded = [ex.awarded for ex in examples if ex.awarded is not None]
    n_questions = len(set(questions))
    return {
        "n_rows": len(examples),
        "n_valid": sum(ex.is_valid for ex in examples),
        "n_invalid": sum(not ex.is_valid for ex in examples),
        "n_questions": n_questions,
        "rows_per_question": round(len(examples) / n_questions, 2) if n_questions else 0,
        "max_marks": dict(sorted(Counter(ex.max_marks for ex in examples).items())),
        "awarded": dict(sorted(Counter(awarded).items())),
        "most_repeated": Counter(questions).most_common(5),
    }


def feature_text(example: Example) -> str:
    return f"Question: {example.question}\nStudent Answer: {example.student_answer}"
