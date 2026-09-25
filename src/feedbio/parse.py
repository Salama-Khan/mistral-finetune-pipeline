from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class UserFields:
    question: str
    max_marks: int
    student_answer: str


@dataclass(frozen=True)
class AssistantFields:
    awarded: int
    max_marks: int
    feedback: str
    tip: str


USER_RE = re.compile(
    r"Question:\s*(?P<question>.*?)\s*"
    r"Max Marks:\s*(?P<max_marks>\d+)\s*"
    r"Student Answer:\s*(?P<student_answer>.*)\s*",
    re.DOTALL,
)

MARKS_RE = re.compile(r"Marks:\s*(?P<awarded>\d+)\s*/\s*(?P<max_marks>\d+)", re.IGNORECASE)
FEEDBACK_RE = re.compile(r"Feedback:\s*(.*?)(?=\n\s*Tip:|\Z)", re.DOTALL | re.IGNORECASE)
TIP_RE = re.compile(r"Tip:\s*(.*)\s*", re.DOTALL | re.IGNORECASE)


def parse_user_prompt(text: str) -> UserFields:
    match = USER_RE.search(text.strip())
    if not match:
        raise ValueError("expected 'Question:', 'Max Marks:', and 'Student Answer:'")
    return UserFields(
        question=match.group("question").strip(),
        max_marks=int(match.group("max_marks")),
        student_answer=match.group("student_answer").strip(),
    )


def parse_assistant_output(text: str) -> AssistantFields:
    marks = MARKS_RE.search(text)
    if not marks:
        raise ValueError("expected 'Marks: X/Y'")
    feedback = FEEDBACK_RE.search(text)
    tip = TIP_RE.search(text)
    return AssistantFields(
        awarded=int(marks.group("awarded")),
        max_marks=int(marks.group("max_marks")),
        feedback=feedback.group(1).strip() if feedback else "",
        tip=tip.group(1).strip() if tip else "",
    )
