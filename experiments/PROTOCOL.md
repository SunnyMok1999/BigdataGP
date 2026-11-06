# Movie sentiment testing and fine tuning

The primary metric is five-class macro F1. Accuracy, class-0 recall, ordinal
mean absolute error, confusion matrices, log loss, and calibration diagnostics
are secondary. Data are the supplied 511 training and 341 test reviews. The
existing report already contains an earlier baseline test result; the test is
held out from every new selection decision here.

## Training-only comparisons

- A TF-IDF logistic-regression comparator tunes 48 settings using five training
  folds. Its reported selected-setting CV score has selection optimism.
- Frozen MiniLM, BGE-small, BERT-base and DistilBERT use raw text, official
  MiniLM mean/BGE CLS pooling (generic mean pooling for BERT/DistilBERT), unit
  normalization and length 128. Logistic regression is tuned in three inner
  folds inside five outer folds. Selecting an encoder family from the outer
  results can still introduce selection optimism.
- Four projection-student procedures use five outer folds. A separate 20%
  inner holdout chooses epochs by negative log likelihood; outer training is
  refitted from scratch for that epoch count. No student training target uses
  an outer-held-out label. Teacher targets inject labels deliberately and are
  never presented as evidence of predictive performance.
- Nine encoder procedures test MiniLM cross entropy, lower learning rate,
  last-two-layer tuning, supervised contrastive loss, teacher cosine loss,
  expected-sentiment ordinal loss, in-batch ordinal hard-negative triplet loss,
  the combined losses, and BGE cross entropy. Five outer folds use a separate
  20% inner holdout to choose epochs by NLL, followed by a fresh outer refit.
  Hyperparameters are fixed before test evaluation. Auxiliary weights are
  documented in encoder_finetuning.py. Encoder LR is 2e-5 (5e-6 in its
  ablation); heads use 1e-3. Maximum ten epochs, patience three, first-epoch
  warmup, then constant learning rate. Final fitting uses the median selected
  epoch count from the five inner holdouts.
- The globally nearest differently labeled training examples exported as CSV
  are reviewable audit examples. Actual hard-negative training mines within
  each minibatch in the current encoder space. Dataset labels establish
  class differences; pairs are not human-verified semantic contradictions.

Outer folds are StratifiedKFold(5, shuffle=True, random_state=42). Inner encoder
and student splits use stratified 20% holdouts with seed 42+fold. All learned
preprocessing stays within its appropriate training partition. Pretrained
checkpoint provenance is recorded; absence of any benchmark material from
large external pretraining corpora cannot be established from the local split.

## Final selection and evaluation

Before evaluating new test performance, save a selection_lock.json based only
on training OOF results. Select one encoder configuration, the best frozen
encoder, and the best projection student by OOF macro F1. Also test a
training-OOF-selected convex blend of the selected encoder and best frozen
classifier; temperature scaling is fitted on training OOF probabilities by
NLL. Calibration and blend selection scores are tuning diagnostics.

Final test evaluation is one batch after model fitting and selection are
complete. Prespecified comparators are the original frozen MiniLM+LR,
training-selected frozen encoder+LR, selected projection student, selected
TF-IDF baseline, selected tuned encoder, and selected calibrated blend.
The primary deployed procedure is chosen by training OOF macro F1, never by
test ranking. Show all comparator test scores without subsequent tuning.
Use a paired, class-stratified bootstrap for test macro-F1 differences and
confidence intervals; these quantify this finite test set, not every source
of training randomness. Do not claim significance from tiny validation gains
or cleaner label-injected clusters.

## Execution constraints

All pretrained checkpoints load locally with local_files_only=True. No paid
service or external training is used. PyTorch's MPS device requires execution
outside the desktop sandbox; this local GPU execution was approved. The CPU
inference path in this PyTorch build segfaulted during batched extraction,
so model extraction and encoder training use MPS. Classifiers and small
projection students run on CPU. No packages are installed or upgraded.
