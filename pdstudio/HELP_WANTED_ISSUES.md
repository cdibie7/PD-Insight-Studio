# Issues to open on the repo
---

## Issue 1

**Title:** Add grouped cross-validation so repeated measures from one participant cannot straddle a train/test split

**Labels:** `help wanted` `good first issue` `modelling`

**Body:**

`modeling.py` currently splits rows without regard to which participant they came from. The dataset is repeated-measures: one participant contributes several sessions. If some of a participant's sessions land in train and others in test, the model can learn the participant rather than the condition, and the reported accuracy will be optimistic. On tapping data this effect is large, not marginal.

**What needs doing**

- Use `GroupKFold` or `StratifiedGroupKFold` from scikit-learn, grouped on the participant ID column.
- The participant ID column is already known to the app: it is mapped to a role on the Configure tab.
- If no participant ID has been mapped, the model should refuse to run rather than silently falling back to a random split, and should say why.

**Why it matters**

This is the single most likely reason a number produced by the Predict tab would fail to replicate. Fixing it makes every result the tool produces more trustworthy.

**Good starting point if you are new to the repo:** you only need to read `modeling.py`, which is about 250 lines.

---

## Issue 2

**Title:** Report calibration and a confusion matrix, not just headline accuracy

**Labels:** `help wanted` `modelling`

**Body:**

The Predict tab reports headline performance. On imbalanced data that is misleading: a classifier that predicts the majority class every time can look respectable.

**What needs doing**

- Add a confusion matrix to the results output.
- Add per-class precision, recall and F1.
- Add a calibration curve, or at minimum a Brier score, so a user can see whether the predicted probabilities mean anything.
- Add balanced accuracy alongside raw accuracy.
- Surface the class balance of the input so the user sees what they are working with before they trust a number.

**Why it matters**

The tool is aimed at researchers who are not primarily modellers. It should make the honest metrics the visible ones.

---

## Issue 3

**Title:** Replace built-in feature importance with permutation importance and optional SHAP

**Labels:** `help wanted` `modelling` `interpretability`

**Body:**

Feature importance currently uses the estimator's built-in attribute. For tree ensembles this is impurity-based, which is biased toward high-cardinality and continuous features, so the ranking can be actively misleading.

**What needs doing**

- Add `sklearn.inspection.permutation_importance` computed on held-out data, and make it the default shown.
- Optionally add SHAP behind a feature flag, since it is a heavy dependency and should not become a hard requirement.
- Where importances are displayed, note which method produced them.

---

## Issue 4

**Title:** Add tests for the QC and rules modules

**Labels:** `help wanted` `good first issue` `testing`

**Body:**

There are currently no tests. `qc.py` and `rules.py` are the highest-value place to start, because they decide which rows get excluded from every downstream result. A silent regression there corrupts everything after it, which is exactly the failure mode this project was built to prevent.

**What needs doing**

- Add `pytest` as a dev dependency.
- Cover `qc.py`: each flagging rule triggers on a row that should fail, does not trigger on a row that should pass, and the recorded reason string matches the rule that fired.
- Cover `rules.py`: derived-variable rules evaluate correctly, and a malformed rule raises rather than silently producing nulls.
- `sample_synthetic_tapping.csv` in the repo root is a ready-made fixture, and it deliberately contains rows that fail QC.

---

## Issue 5

**Title:** Document the expected input schema and fail clearly when a required role is unmapped

**Labels:** `help wanted` `good first issue` `documentation`

**Body:**

A new user does not know what shape of CSV the app expects until they load one and something goes wrong.

**What needs doing**

- Document the column roles the app understands (participant ID, session, timepoint, group, outcome), which are required for which tab, and what types are expected.
- When a tab is opened without the roles it needs, show which role is missing and where to set it, rather than erroring or producing an empty result.

**Why it matters**

This is the first wall anyone hits, including anyone evaluating the project. It is also a genuinely easy first contribution.
