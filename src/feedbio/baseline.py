from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from .data import Example
from .features import feature_text


def make_pipeline() -> Pipeline:
    # Ridge over a classifier because the label is a mark, not a class that
    # means the same thing on a 1-mark question and a 6-mark question.
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=4000),
            ),
            ("reg", Ridge(alpha=1.0)),
        ]
    )


def fit_baseline(train: list[Example]) -> Pipeline:
    usable = [ex for ex in train if ex.awarded is not None]
    if len(usable) < 4:
        raise ValueError("need at least 4 labelled examples to fit")
    texts = [feature_text(ex) for ex in usable]
    y = [ex.awarded for ex in usable]
    pipe = make_pipeline()
    pipe.fit(texts, y)
    return pipe


def predict_marks(pipe: Pipeline, examples: list[Example]) -> list[int]:
    texts = [feature_text(ex) for ex in examples]
    raw = pipe.predict(texts)
    preds = []
    for ex, value in zip(examples, raw):
        rounded = round(float(value))
        preds.append(max(0, min(ex.max_marks, rounded)))
    return preds
