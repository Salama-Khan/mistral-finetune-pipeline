# Baseline eval (TF-IDF + Ridge)

## Data

- rows: 128
- valid: 128
- invalid: 0
- unique questions: 27
- rows per question: 4.74
- max-mark counts: {1: 39, 2: 58, 3: 16, 4: 10, 6: 5}
- awarded-mark counts: {0: 54, 1: 43, 2: 18, 3: 8, 4: 4, 5: 1}

Most repeated questions:

- 20× Give two similarities between prokaryotic cells and eukaryotic cells.
- 11× What is the function of the nucleus?
- 11× Give two advantages of using an electron microscope instead of a light microscope.
- 10× Explain why the red blood cell bursts but the plant cell does not burst when placed in water.
- 9× Describe three differences between the processes of mitosis and meiosis.

## Split

Question-level holdout, seed=42, test_size=0.2. 93 train rows (22 questions), 35 test rows (5 questions).

## Metrics

- n scored: 35
- MAE: 0.829
- exact match: 34.3%
- within 1 mark: 82.9%
- format parse rate: 100.0%

## Largest errors

- gold 0/2, pred 2: Explain one way in which the root hair cell is adapted to take up water. — active transport
- gold 0/2, pred 2: Explain why the red blood cell bursts but the plant cell does not burst when placed in water. — Red blood cell is smaller than plant cell so it cant hold as much water
- gold 0/2, pred 2: Explain why the red blood cell bursts but the plant cell does not burst when placed in water. — red blood cell is weaker than plant cell
- gold 0/2, pred 2: Explain why the red blood cell bursts but the plant cell does not burst when placed in water. — Water moves in and the plant cell stays strong.
- gold 0/2, pred 2: Explain why the red blood cell bursts but the plant cell does not burst when placed in water. — Cells are made of cellulose which helps them stay strong.
- gold 0/2, pred 2: Give two advantages of using an electron microscope instead of a light microscope. — You can see cells with more detail, but they are more powerful than normal microscopes.
- gold 0/2, pred 1: Explain one way in which the root hair cell is adapted to take up water. — Root hair cells have hairs which are good at taking in water
- gold 0/2, pred 1: Explain one way in which the root hair cell is adapted to take up water. — Root hair cells have hairs which are good at taking in water

## Notes

Question-level holdout is a harder test than a random row split: the model
has to mark questions it has never seen. A bag-of-n-grams baseline is not a
good examiner, and these numbers are here as a comparison point, not a claim
that TF-IDF can replace a teacher.
