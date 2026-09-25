from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .parse import parse_assistant_output, parse_user_prompt

EXPECTED_ROLES = ("system", "user", "assistant")


@dataclass
class Example:
    question: str
    max_marks: int
    student_answer: str
    awarded: int | None
    feedback: str | None
    tip: str | None
    messages: list[dict]
    issues: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
    return records


def example_from_record(record: dict) -> Example:
    issues: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("record needs a messages list with system/user/assistant")

    roles = tuple(m.get("role") for m in messages)
    if roles != EXPECTED_ROLES:
        issues.append(f"roles {roles} != {EXPECTED_ROLES}")

    by_role = {m.get("role"): m.get("content", "") for m in messages}

    try:
        user = parse_user_prompt(str(by_role.get("user", "")))
    except ValueError as exc:
        issues.append(str(exc))
        user = None

    try:
        assistant = parse_assistant_output(str(by_role.get("assistant", "")))
    except ValueError as exc:
        issues.append(str(exc))
        assistant = None

    question = user.question if user else ""
    max_marks = user.max_marks if user else 0
    student_answer = user.student_answer if user else ""
    awarded = assistant.awarded if assistant else None
    feedback = assistant.feedback if assistant else None
    tip = assistant.tip if assistant else None

    if user and assistant and assistant.max_marks != user.max_marks:
        issues.append(
            f"assistant denom {assistant.max_marks} != user max {user.max_marks}"
        )
    if awarded is not None and awarded < 0:
        issues.append(f"awarded {awarded} is negative")
    if awarded is not None and awarded > max_marks:
        issues.append(f"awarded {awarded} exceeds max {max_marks}")

    return Example(
        question=question,
        max_marks=max_marks,
        student_answer=student_answer,
        awarded=awarded,
        feedback=feedback,
        tip=tip,
        messages=messages,
        issues=issues,
    )


def load_examples(path: str | Path) -> list[Example]:
    return [example_from_record(record) for record in load_jsonl(path)]


def split_by_question(
    examples: list[Example],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[Example], list[Example]]:
    """Hold out whole questions.

    A row-wise split leaks: some questions appear 20 times with slightly
    different student answers, so the model would mostly memorise the question.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    questions = sorted({ex.question for ex in examples if ex.question})
    rng = random.Random(seed)
    rng.shuffle(questions)

    n_test = max(1, round(len(questions) * test_size))
    test_questions = set(questions[:n_test])

    train = [ex for ex in examples if ex.question not in test_questions]
    test = [ex for ex in examples if ex.question in test_questions]
    if not train or not test:
        raise ValueError("split produced an empty side; try a different seed")
    return train, test
