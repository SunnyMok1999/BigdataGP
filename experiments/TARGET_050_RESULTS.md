# BigDataGP: Search for 0.50 five-class sentiment

**Dataset:** 511 labeled training reviews, five sentiment labels (0,1,2,3,4).
The separate 341-review `test.csv` has **not** been used for model tuning,
scoring, or selection in this branch.

## Strongest completed result

A pretrained Amazon-review 5-star RoBERTa encoder
(`Nav772/roberta-amazon-reviews-5star`) supplied **fixed pretrained 768-dimensional
CLS sentence features**. A fold-trained `StandardScaler` and `SVC(kernel="rbf")`
with inner 3-fold cross-validated C, gamma and balanced class weighting obtained:

| Metric | Training-only OOF |
|---|---:|
| Accuracy | **0.5010 (50.10%)** |
| Macro-F1 | **0.4971** |

The 0.50 accuracy goal is achieved on this training-only CV estimate.
Macro-F1 is **not yet 0.50**. This estimate is *not* the hidden test score.

The exact training code is in `experiments/target_050_amazon.py` and
evidence with fold hyperparameters and predictions is in:
https://github.com/SunnyMok1999/BigdataGP/actions/runs/42241414807

## Other results

- Amazon RoBERTa unscaled hidden-vector RBF SVM (previous experiment):
  macro-F1 **0.4874**, accuracy 0.4795.
- Decontaminated public SST5 **extra supervised labeled training data**:
  macro-F1 **0.4822**, accuracy 0.4834; may be disallowed in this course.
  See separate branch `experiment/target-macro-f1-050`.
- Predeclared three-way hard-label majority vote over Amazon, decontaminated
  SST5 and BGE: macro-F1 0.4873, accuracy 0.4873.
- Nested BGE-only RBF SVC: macro-F1 0.4469.
- Nested BGE linear logistic regression: macro-F1 0.4334.
- Nested BGE ordinal classifier: macro-F1 0.4299.
- Nested BGE+MiniLM RBF: macro-F1 0.4374.
- Nested BGE+BERT RBF: macro-F1 0.4365.
- Nested four-encoder logistic: macro-F1 0.4028.
- Nested four-encoder RBF: macro-F1 0.4102.
- Nested word+character TFIDF SVM: macro-F1 0.2458.
- Nested word+character TFIDF+BGE SVM: macro-F1 0.4123.
- Amazon RoBERTa unscaled RBF tuned kernel: macro-F1 0.4818.
- Amazon RoBERTa polynomial kernel degree 2: macro-F1 0.4741.
- Amazon RoBERTa polynomial kernel degree 3: macro-F1 0.4766.
- Amazon RoBERTa hidden-vector linear SVC: macro-F1 0.4500.
- Amazon RoBERTa hidden-vector logistic regression: macro-F1 0.4509.

## Evidence

- Cached embeddings / lexical [nested-CV run](https://github.com/SunnyMok1999/BigdataGP/actions/runs/42240203903)
- Amazon kernel [nested-CV run](https://github.com/SunnyMok1999/BigdataGP/actions/runs/42241414807)
- Fixed vote [OOF run](https://github.com/SunnyMok1999/BigdataGP/actions/runs/42240666570)
- Fine-grained normalization and class-weight [nested CV](https://github.com/SunnyMok1999/BigdataGP/actions/runs/42241891902)

## Integrity / caution

1. Five outer stratified folds, with hyperparameters trained only in inner
   stratified folds, produce label-free OOF predictions for each candidate.
2. Comparing many candidates using those outer-fold scores introduces
   *model-family selection optimism*. Even 0.5010 accuracy should not be
   treated as a guaranteed independent-test improvement.
3. All pretrained sentence encoders were trained on external corpora.
   Verify that the course permits external pretraining, and document
   provenance. **Extra supervised SST5 training** is separately flagged
   because it uses task-specific external labels.
4. Freeze the final procedure before evaluating the held-out 341 rows.
   Do not optimize thresholds or model families against hidden test labels.

## Fine-grained refinements (completed)

All three further training-only nested-CV runs passed:

| Amazon RoBERTa 768D classifier | OOF accuracy | OOF Macro-F1 |
|---|---:|---:|
| StandardScaler + RBF SVM (original best) | **0.5010** | **0.4971** |
| StandardScaler + L2 + RBF SVM | 0.4892 | 0.4957 |
| StandardScaler + finer C/gamma RBF SVM | 0.4873 | 0.4953 |
| StandardScaler + alternative class weights | 0.4834 | 0.4933 |

Evidence: https://github.com/SunnyMok1999/BigdataGP/actions/runs/42241891902

**Final outcome in this search:** 50% training OOF accuracy achieved;
Macro-F1 remains below 0.50. Repeated model-family comparison can overfit the
shared outer folds, so these are exploratory results, not a guaranteed
independent-test score. The 341-review test labels remain untouched.
