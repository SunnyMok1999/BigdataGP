# Closed-loop sentiment research workflow

This addition extends the existing `experiments/PROTOCOL.md`; it does **not** change
previous train-only results, the frozen selection lock, or the 341-row `test.csv`.

## Current evidence and baseline

The repository already contains genuine cached pretrained MiniLM and BGE-small
features and fine-tuning CV runs. The nested training-only **BGE-small + logistic
regression** result is 0.4188 accuracy / 0.4148 macro-F1 in
`experiments/results/frozen/comparison.csv`. The new workflows below have **not**
been benchmarked yet against that BGE result. Do not compare previously
reported ChatGPT figures from different splits as if they were matched runs.

## Install and verify

From the repository root:

```bash
python -m pip install -r experiments/requirements-workflow.txt
python -m pytest -q tests/test_workflows.py
```

The frozen-feature active-learning simulation uses only NumPy, pandas and
scikit-learn. Encoder fine-tuning requires PyTorch, Transformers, and a local
Hugging Face checkpoint (see existing `encoder_finetuning.py`). In this
repository the official MiniLM model is
`sentence-transformers/all-MiniLM-L6-v2`, BGE is
`BAAI/bge-small-en-v1.5`. Raw text, rather than cleaned/stemmed text, is used
for pretrained encoders.

## A. Compare active-learning acquisition policies

```bash
python experiments/active_learning_loop.py simulate \
  --seeds 42 123 2026 --per-class 10 --batch 20 --rounds 8 \
  --policies random entropy margin adjacent entropy_diversity \
  --output experiments/results/active_learning_new
```

Defaults use the existing BGE-small 511-row frozen embedding cache and matching
`bge_small_train_rows.csv`. The script refuses mismatched row order, requires
unique PhraseId values and rejects duplicated SentenceId values in simulation.
It splits **train.csv only** into an 80% acquisition pool and a 20% offline
evaluation split. All policies share the same split and 50-label stratified
initialization per seed. Such stratified initialization requires access to
labels in simulation: it is not available in real cold-start labeling.
Gold pool labels are revealed to the learner **only after** query selection.

Outputs: `curves.csv`, `summary.csv`, `acquisitions.csv`. Scores include
accuracy, macro-F1 and recall for each of five classes. The `adjacent` policy
prioritizes uncertainty between classes 0/1, 1/2, 2/3 and 3/4 and adds a
modest bonus for boundaries 0/1 and 3/4. Compare it to random sampling using
**matched labeled-example counts**, not absolute training speed. Models are
trained from scratch on acquired labels each round, using a fixed C=1.0.

## B. Human-in-the-loop queue

Provide an **unlabeled** CSV containing `PhraseId,Phrase`, a matching
precomputed embeddings `.npy`, and a `--rows-csv` with matching PhraseId order.
Do NOT include hidden truth in the pool CSV. Human annotations must be a
separate `PhraseId,Sentiment` CSV (labels 0..4).

```bash
python experiments/active_learning_loop.py human-next \
  --csv data/new_unlabeled.csv \
  --embeddings data/new_embeddings.npy \
  --rows-csv data/new_embedding_rows.csv \
  --annotations data/human_labels.csv --policy adjacent --batch 10 \
  --output experiments/results/human_queue
```

The first queue is random until all five classes have labeled support.
Fill labels in `annotation_batch.csv`, append verified labels to your separate
`human_labels.csv`, then run `human-next` again. Predictions are deliberately
not shown to annotators. The CLI is stateless: re-running it without updating
the annotations can repeat a queue. This version requires embeddings for
unseen texts to be computed separately with an available pretrained encoder.

## C. Mine, audit and approve hard negatives

```bash
python experiments/hard_negative_review.py mine \
  --out experiments/results/hard_negative_review
```

This uses **training-only out-of-fold** BGE predictions and the original
pretrained BGE vectors. Outputs:

- `label_audit_REVIEW.csv`: loss, entropy, predictions, and possible noisy
  labels for review. High-confidence disagreement is only a *flag*, not proof.
- `adjacent_triplets_REVIEW.csv`: nearest same-class positives and adjacent
  different-class negatives, prioritized by model difficulty. Even though the
  labels differ, the *intensity contrast* has **not** been human verified.
- `augmentation_REVIEW.csv`: paraphrase instructions and an empty
  `augmented_text` column. Do not assign the original label blindly to
  generated text; check negation, polarity, and intensity.

Every approval field starts false. After reviewing, export with:

```bash
python experiments/hard_negative_review.py export-approved \
  --aug-file experiments/results/hard_negative_review/augmentation_REVIEW.csv \
  --triplet-file experiments/results/hard_negative_review/adjacent_triplets_REVIEW.csv \
  --out experiments/results/hard_negative_review/approved
```

The exporter rejects missing IDs, incorrect labels, non-adjacent negatives,
and insufficient paraphrases. It does **not** magically certify semantics.
Important: the globally mined review queue is for **human audit**. Its
label-derived pair choices must not be injected into a held-out fold.
Approved augmentation/pair exports are **not automatically fed into the current
fine-tuning code**; this is deliberate until labels and fold-safe sampling
have been verified.

## D. Train with the new adjacent-class objective

```bash
# First smoke-check loss gradients with pytest. Then on your local GPU:
python experiments/encoder_finetuning.py cv \
  --configs minilm_adjacent_v2 --device mps
```

Use `--device cuda` on NVIDIA machines, or `--device cpu` if supported.

The added configuration combines cross-entropy with three lightweight
regularizers: expected probability on the wrong adjacent labels (weight
0.15), hardest adjacent-class triplet margin (weight 0.10), and
adjacent-centroid separation (weight 0.05). The 0/1 and 3/4 boundaries use
higher proposed cosine margins than 1/2 and 2/3. These weights are **initial
hypotheses**, not optimized hyperparameters.

The encoder CV uses the existing five outer folds and separate inner
epoch-selection holdout. Do not judge new methods from training-cluster plots.
Track class-0 and class-4 recall, macro-F1, NLL and confusion matrices;
compare all methods on matched training partitions. Selecting the best among
many variants on these CV folds introduces further selection optimism.

The existing `select_and_evaluate.py` selection set is intentionally **not
modified**, because its previous model-selection protocol should not be
silently changed after inspection. No new 341-review test results are claimed.

## What remains

A separate *fold-safe*, human-reviewed text-augmentation training pipeline,
true active learning on new unlabeled reviews with refreshed encoder embeddings,
and a properly nested comparison of selection/calibration/ensemble alternatives
are next research stages. The review files are not assumed to be verified
training data merely because they have been generated.


## Experiment A — 20-seed acquisition comparison (new)

The repository now supports `hybrid` (65% adjacent-boundary uncertainty,
35% global entropy, each rank-normalized) and `balanced_hybrid`
(85% hybrid uncertainty plus 15% expected inverse-frequency class priority).
The latter uses **only the labels already acquired**, never candidate gold
labels. Both are hypotheses, not universally better query policies.

Run the same benchmark on the stronger pretrained **BGE-small frozen**
embeddings with 20 predefined random seeds:

```bash
python experiments/active_learning_loop.py simulate \
  --policies random entropy adjacent hybrid balanced_hybrid \
  --seeds 42 123 2026 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 \
  --per-class 10 --batch 20 --rounds 8 \
  --output experiments/results/active_learning_BGE_20seeds

python experiments/report_AB.py \
  --curves experiments/results/active_learning_BGE_20seeds/curves.csv
```

Read `summary.csv`, `paired_comparison.csv`, `class_recall.csv`,
`learning_curve_auc.csv`, and the consolidated report. Use the **paired
per-seed** gain against random and average macro-F1 across all budgets.
The normal 95% intervals are approximate and do not cover method-selection
optimism. Do not use the 341-row test set for method selection.

A separate 20-seed exploratory run using saved **MiniLM** embeddings,
with the exact seed list shown above, achieved the following final
(210-label) macro-F1: balanced hybrid **0.3123**, hybrid 0.3114,
adjacent 0.3046, random 0.3023 and entropy 0.3007. The balanced
hybrid minus random difference was +0.0100 with an approximate normal
95% interval [-0.0079, 0.0279]; hybrid minus random was +0.0091
[-0.0097, 0.0278]. Neither establishes a reliable advantage.
On mean macro-F1 area under the learning curve, hybrid 0.2933 led
balanced hybrid 0.2915, entropy 0.2892, adjacent 0.2889 and random
0.2880. These are **MiniLM** frozen-feature results: the user's
earlier three-seed finding used BGE-small and requires a directly
matched 20-seed BGE replication.

## Experiment B — controlled pretrained-encoder ablations (new)

`minilm_adjacent_v2` is the existing combined adjacent loss.
The new encoder configurations are:

- `minilm_adjacent_light`: lower objective weights and smaller margins.
- `minilm_adjacent_strong`: larger weights and margins.
- `minilm_adjacent_no_triplet`: ablate the triplet loss.
- `bge_adjacent_v2`: apply the reference adjacent loss to pretrained BGE-small.
- `bge_ce`: cross-entropy baseline on the same BGE architecture.

Run only on a machine with the locally cached model checkpoints:

```bash
python experiments/encoder_finetuning.py cv \
  --configs minilm_adjacent_light,minilm_adjacent_strong,minilm_adjacent_no_triplet,bge_ce,bge_adjacent_v2 \
  --device mps

python experiments/report_AB.py \
  --curves experiments/results/active_learning_BGE_20seeds/curves.csv
```

The existing `minilm_ce`, `minilm_supcon`, `minilm_ordinal`, and
`minilm_adjacent_v2` summaries are included when available. This is
**full pretrained encoder fine-tuning**, unlike the frozen-feature student.

The initial 5-fold **frozen MiniLM 384D -> 16D projection** ablation,
conducted separately without pretrained weights, scored macro-F1:
CE only 0.2953; light 0.2916; default 0.2944; strong 0.2756.
It shows that stronger margins alone can hurt generalization, but
**it cannot establish whether BGE or MiniLM encoder fine-tuning benefits**.
The source and detailed results of this exploratory projection ablation
are distributed separately, not represented as finished encoder results.

### Validation and limitations

- All CV runs keep the 341-row held-out test untouched.
- The user must run MPS fine-tuning on their Mac: no local pretrained model
  checkpoints or `transformers` are available in this sandbox.
- The class-balanced query does **not** see candidate gold labels.
- True augmentation requires a human label-consistency gate and strictly
  train-fold-local data inclusion; generated queues alone are not proof.
- Compare both mean macro-F1 and recall for **every class**. Extra losses
  should be rejected if they improve embeddings visually but hurt macro-F1.
