from feedbio.baseline import fit_baseline, predict_marks
from feedbio.data import example_from_record
from feedbio.evaluate import evaluate_marks


def _labelled(question, answer, awarded, max_marks=2):
    return example_from_record(
        {
            "messages": [
                {"role": "system", "content": "examiner"},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"Max Marks: {max_marks}\n"
                        f"Student Answer: {answer}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"Marks: {awarded}/{max_marks}\nFeedback: x\nTip: y",
                },
            ]
        }
    )


def test_baseline_fits_and_returns_one_pred_per_row():
    train = [
        _labelled("What is the nucleus?", "contains DNA", 1, 1),
        _labelled("What is the nucleus?", "no idea", 0, 1),
        _labelled("What are ribosomes?", "protein synthesis", 1, 1),
        _labelled("What are ribosomes?", "store water", 0, 1),
        _labelled("What is mitosis?", "cell division for growth", 2, 2),
        _labelled("What is mitosis?", "photosynthesis", 0, 2),
        _labelled("What is meiosis?", "makes gametes", 1, 2),
        _labelled("What is meiosis?", "a bone", 0, 2),
    ]
    pipe = fit_baseline(train)
    preds = predict_marks(pipe, train[:3])
    assert len(preds) == 3
    assert all(0 <= p <= ex.max_marks for p, ex in zip(preds, train[:3]))

    result = evaluate_marks(train[:3], preds)
    assert result.n == 3
    assert result.mae >= 0
