from __future__ import annotations

from dataclasses import dataclass, field

from .data import Example
from .parse import parse_assistant_output


@dataclass
class EvalResult:
    n: int
    mae: float
    exact_match: float
    within_1: float
    format_parse_rate: float
    errors: list[dict] = field(default_factory=list)


def evaluate_marks(
    examples: list[Example],
    preds: list[int | None],
    top_errors: int = 8,
) -> EvalResult:
    if len(examples) != len(preds):
        raise ValueError("examples and preds must be the same length")

    abs_err = []
    exact = 0
    within = 0
    scored = 0
    errors = []

    for ex, pred in zip(examples, preds):
        if ex.awarded is None or pred is None:
            continue
        scored += 1
        err = abs(pred - ex.awarded)
        abs_err.append(err)
        if err == 0:
            exact += 1
        if err <= 1:
            within += 1
        if err > 0:
            errors.append(
                {
                    "question": ex.question,
                    "student_answer": ex.student_answer,
                    "gold": ex.awarded,
                    "pred": pred,
                    "max_marks": ex.max_marks,
                    "abs_err": err,
                }
            )

    if scored == 0:
        raise ValueError("no scored examples")

    errors.sort(key=lambda row: (-row["abs_err"], row["question"]))
    return EvalResult(
        n=scored,
        mae=sum(abs_err) / scored,
        exact_match=exact / scored,
        within_1=within / scored,
        format_parse_rate=1.0,
        errors=errors[:top_errors],
    )


def evaluate_generations(examples: list[Example], texts: list[str]) -> EvalResult:
    preds: list[int | None] = []
    parsed = 0
    for text in texts:
        try:
            preds.append(parse_assistant_output(text).awarded)
            parsed += 1
        except ValueError:
            preds.append(None)

    result = evaluate_marks(examples, preds)
    result.format_parse_rate = parsed / len(texts) if texts else 0.0
    return result


def format_report(
    title: str,
    stats: dict,
    split_note: str,
    result: EvalResult,
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Data",
        "",
        f"- rows: {stats['n_rows']}",
        f"- valid: {stats['n_valid']}",
        f"- invalid: {stats['n_invalid']}",
        f"- unique questions: {stats['n_questions']}",
        f"- rows per question: {stats['rows_per_question']}",
        f"- max-mark counts: {stats['max_marks']}",
        f"- awarded-mark counts: {stats['awarded']}",
        "",
        "Most repeated questions:",
        "",
    ]
    for question, count in stats["most_repeated"]:
        lines.append(f"- {count}× {question}")

    lines += [
        "",
        "## Split",
        "",
        split_note,
        "",
        "## Metrics",
        "",
        f"- n scored: {result.n}",
        f"- MAE: {result.mae:.3f}",
        f"- exact match: {result.exact_match:.1%}",
        f"- within 1 mark: {result.within_1:.1%}",
        f"- format parse rate: {result.format_parse_rate:.1%}",
        "",
        "## Largest errors",
        "",
    ]
    if not result.errors:
        lines.append("None.")
    else:
        for row in result.errors:
            answer = row["student_answer"].replace("\n", " ")
            if len(answer) > 140:
                answer = answer[:137] + "..."
            lines.append(
                f"- gold {row['gold']}/{row['max_marks']}, pred {row['pred']}: "
                f"{row['question']} — {answer}"
            )
    lines += [
        "",
        "## Notes",
        "",
        "Question-level holdout is a harder test than a random row split: the model",
        "has to mark questions it has never seen. A bag-of-n-grams baseline is not a",
        "good examiner, and these numbers are here as a comparison point, not a claim",
        "that TF-IDF can replace a teacher.",
        "",
    ]
    return "\n".join(lines)
