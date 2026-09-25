from feedbio.data import example_from_record, split_by_question


def _record(question="What is a stem cell?", max_marks=2, answer="unspecialised cell", marks="1/2"):
    return {
        "messages": [
            {"role": "system", "content": "examiner"},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\nMax Marks: {max_marks}\nStudent Answer: {answer}"
                ),
            },
            {
                "role": "assistant",
                "content": f"Marks: {marks}\nFeedback: ok\nTip: revise",
            },
        ]
    }


def test_example_from_record_ok():
    ex = example_from_record(_record())
    assert ex.is_valid
    assert ex.awarded == 1
    assert ex.max_marks == 2


def test_awarded_over_max_is_invalid():
    ex = example_from_record(_record(max_marks=1, marks="2/1"))
    assert not ex.is_valid
    assert any("exceeds max" in issue for issue in ex.issues)


def test_wrong_roles_are_invalid():
    record = _record()
    record["messages"][0]["role"] = "teacher"
    ex = example_from_record(record)
    assert not ex.is_valid


def test_split_holds_out_whole_questions():
    examples = []
    for q in ("nucleus", "ribosome", "mitosis"):
        for i in range(4):
            examples.append(example_from_record(_record(question=q, answer=f"ans {i}")))

    train, test = split_by_question(examples, test_size=0.33, seed=0)
    train_q = {ex.question for ex in train}
    test_q = {ex.question for ex in test}
    assert train_q.isdisjoint(test_q)
    assert train_q | test_q == {"nucleus", "ribosome", "mitosis"}
