# BigdataGP — Five-Class Movie Review Sentiment

Research code for **DSAI4205** group-project movie-review sentiment classification.
The existing 511-review training set is `train.csv`, with a separate
341-review `test.csv`. The original data cleaning, EDA and baseline notebooks
are preserved.

## Existing experiments

- `experiments/baseline_tuning.py`: TF-IDF logistic regression.
- `experiments/frozen_embeddings.py`: cached MiniLM, BGE-small, BERT-base and
  DistilBERT embeddings with inner tuning / outer evaluation.
- `experiments/encoder_finetuning.py`: pretrained MiniLM/BGE tuning and
  auxiliary loss ablations.
- `experiments/student_distillation.py`: label-coded teacher vectors and
  out-of-fold student generalization checks.
- `experiments/PROTOCOL.md`: training-only selection and test safeguards.

The stored nested-CV BGE-small + logistic-regression run has training OOF
macro-F1 **0.4148** and accuracy **0.4188**
(`experiments/results/frozen/comparison.csv`). These are not scores for
the new methods below.

## New: active learning and adjacent-class research

See **[CLOSED_LOOP_WORKFLOW.md](experiments/CLOSED_LOOP_WORKFLOW.md)** for
commands, assumptions, approval gates and limitations.

- `experiments/active_learning_loop.py`: random, entropy, margin, adjacent,
  and uncertainty+diversity acquisition; offline simulations and human queues.
- `experiments/hard_negative_review.py`: OOF difficulty/noise ranking,
  adjacent hard-negative candidates, and human-reviewed paraphrase exports.
- `experiments/ordinal_losses.py`: extra adjacent-confusion cost, cosine
  triplet margins and prototype separation. The encoder exposes
  `minilm_adjacent_v2` as a new opt-in configuration.
- `tests/test_workflows.py`: gradient, no-gold-data, alignment and human
  approval sanity tests.

**No test-set results or new accuracy improvements are claimed for these
additions.** Review queues are not labels verified by a human, and modified
embedding geometry alone does not imply better unseen-review performance.
