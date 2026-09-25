import pytest

from feedbio.parse import parse_assistant_output, parse_user_prompt


def test_parse_user_prompt():
    fields = parse_user_prompt(
        "Question: What is the function of the nucleus?\n"
        "Max Marks: 1\n"
        "Student Answer: hold dna"
    )
    assert fields.question == "What is the function of the nucleus?"
    assert fields.max_marks == 1
    assert fields.student_answer == "hold dna"


def test_parse_user_prompt_rejects_missing_fields():
    with pytest.raises(ValueError):
        parse_user_prompt("just a free-text answer")


def test_parse_assistant_output():
    fields = parse_assistant_output(
        "Marks: 1/2\nFeedback: Partial credit for naming DNA.\nTip: Mention control of the cell."
    )
    assert fields.awarded == 1
    assert fields.max_marks == 2
    assert "Partial credit" in fields.feedback
    assert "control" in fields.tip


def test_parse_assistant_output_rejects_missing_marks():
    with pytest.raises(ValueError):
        parse_assistant_output("Feedback: good try\nTip: revise organelles")
