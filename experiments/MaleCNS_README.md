# MaleCNS subgraph sentiment classifier (exploratory)

This is a **train-only** experiment for the five sentiment labels in the
511-row `train.csv`. Do not use `test.csv` to select graph sizes, training
epochs, seeds, or other hyperparameters. The existing 341-review test remains
held out until the final procedure is locked.

## Source of biological graph

`malecns_data/graph.json` is a measured **80-neuron, 1,296-edge** induced
MaleCNS v1.0 circuit originally constructed for the open-source
[cobanov/flyjump](https://github.com/cobanov/flyjump) research project.
It contains original neuron body IDs, measured directed edges with contact
counts, and annotations. It is derived from [Janelia's MaleCNS v1.0]
(https://male-cns.janelia.org/download/) and redistributed under CC BY 4.0.
See `malecns_data/manifest.json` and `malecns_data/NOTICE.md`.

This circuit is **selected around a fly sensorimotor pathway**, not around
language. The graph is therefore an exploratory inductive bias, not a
physiological language network, and does not model real fruit-fly cognition.

## Architecture

- Raw review text is tokenized into deterministic hash IDs, with no
  label-dependent vocabulary fitting.
- A 48-dimensional learned word-embedding layer feeds a recurrent network.
- The recurrent weight matrix has exactly the binary directed connection
  pattern of the selected 32-, 64-, or 80-neuron MaleCNS *induced subgraph*.
- Trainable input projection, masked recurrent weights, and five-class
  classifier are updated by cross entropy. Final and mean hidden states are
  pooled for the prediction.
- Measured synapse counts and neurotransmitter signs are **not** interpreted
  as fixed biological simulation parameters; only the edges are reused.
- Fixed-epoch training and 5-fold stratified OOF estimates avoid using an outer
  fold's labels for early stopping. No held-out test labels are read.

## Controls

`biological`: source directed edges.

`degree_rewired`: directed double-edge switches preserve each node's
in-degree and out-degree as well as total edges.

`edge_random`: same node and edge counts, but no degree matching.

All arms have identical trainable model architecture and optimizers and use
the same dataset folds and training seeds. A stronger external baseline is
the previously measured **BGE-small + Logistic Regression** training-only
nested-CV score (macro-F1 0.4148; accuracy 0.4188), described in
`experiments/PROTOCOL.md`. Its preprocessing and architecture differ;
compare it as an established performance reference, not a topology ablation.

## Run

From the repository root:

```bash
python -m pip install numpy pandas scikit-learn torch pytest
python -m pytest -q tests/test_malecns.py

# Quick end-to-end smoke check (undertrained, NOT a performance result):
python experiments/malecns_sentiment.py --nodes 80 --folds 3 --epochs 2 \
  --output experiments/results/malecns_smoke

# Main train-only paired comparison:
python experiments/malecns_sentiment.py --nodes 80 --folds 5 --epochs 12 \
  --seeds 42 123 2026 --output experiments/results/malecns_80

# Graph-size ablations (new experiment families, interpret carefully):
python experiments/malecns_sentiment.py --nodes 32 --folds 5 --epochs 12 \
  --seeds 42 123 2026 --output experiments/results/malecns_32
python experiments/malecns_sentiment.py --nodes 64 --folds 5 --epochs 12 \
  --seeds 42 123 2026 --output experiments/results/malecns_64
```

Outputs: `fold_metrics.csv`, `summary.csv`,
`oof_predictions.csv`, `run_config.json`. Report OOF macro-F1 as primary
metric and per-class recall, accuracy and confusion matrix as diagnostics.

## Interpret cautiously

On a 511-row dataset, a substantial graph can overfit. **No accuracy or
macro-F1 improvement is claimed** without running the experiment. Check
consistent positive paired differences against rewired controls over several
random seeds. Changing graph size, epochs or losses based on OOF scores is
model selection and adds optimism. If these experiments inform final selection,
run a new nested evaluation or reserve the untouched test set for one
pre-registered final comparison. A graph effect is not evidence of insect
language understanding.

## BGE-driven recurrent variant

The independent `experiments/malecns_bge.py` experiment reuses the existing
**train-only, pretrained BGE-small sentence embeddings** (shape 511 × 384)
rather than attempting to learn a useful language encoder from 511 labels.
A trainable 384-to-N adapter injects each sentence vector as a *constant drive*
for six recurrent updates of the MaleCNS-masked hidden state. The state is
read out to five sentiment classes. These six iterations are **not token
positions**, and there is no claim that the measured fly subgraph evolved
for language processing. This variant better isolates the choice of recurrent
graph once the NLP representation is fixed.

It also has the same biological, degree-preserving rewired and edge-matched
random controls. It asserts exact `PhraseId` and label alignment before
loading embeddings to prevent using unrelated rows. No `test.csv` is used.

```bash
python -m pytest -q tests/test_malecns_bge.py
python experiments/malecns_bge.py --nodes 80 --epochs 30 --steps 6 \\
  --seeds 42 123 2026 --output experiments/results/malecns_bge_80
```

Compare against the repository's existing nested-CV BGE-small + LR score,
acknowledging that our frozen-epoch configuration and the LR inner-tuning
protocol are not perfectly matched. If the recurrent model does not improve
macro-F1 *and* consistently beat rewired controls across seeds, the
biological graph is not currently helping sentiment classification.

## Comparison to standard RNN and LSTM

`experiments/rnn_lstm_sentiment.py` benchmarks three **token-sequence**
architectures with the same fixed text hashing, learned 48D word embedding,
80-unit hidden size, mean-plus-final-state readout, class-weighted CE, folds,
training epochs, batch size, optimizer and random seeds:

- `malecns`: the original sparse biological recurrent mask.
- `rnn`: ordinary one-layer unidirectional tanh `nn.RNN`.
- `lstm`: one-layer unidirectional `nn.LSTM`, with learned input/forget/output gates.

The plain RNN and LSTM have dense hidden connections and **different recurrent
parameter counts** from MaleCNS, so a performance difference cannot be attributed
to biological topology alone. The LSTM is also more expressive due to gating.
The comparison is intended to ask whether the graph-constrained model can be
competitive with ordinary sequence models under the same training protocol.
For strict graph-topology comparisons, see the degree-rewired / edge-random
controls reported above.

Run from the repo root:

```bash
python -m pytest -q tests/test_malecns.py tests/test_rnn_lstm_sentiment.py
python experiments/rnn_lstm_sentiment.py \\
  --models malecns rnn lstm --folds 5 --hidden-size 80 \\
  --epochs 12 --seeds 42 123 2026 \\
  --output experiments/results/rnn_lstm_comparison
```

This only uses `train.csv` (511 labeled movie-review phrases); `test.csv`
remains untouched. Results include `summary.csv`, `per_seed_metrics.csv`,
`fold_metrics.csv`, `oof_predictions.csv` and `metadata.json`.
Cross-model comparisons use **complete OOF macro-F1 per seed** (then average
across seeds), not pooled predictions across independently trained seeds.
See the [automated experiment](https://github.com/SunnyMok1999/BigdataGP/actions/workflows/rnn_lstm_sentiment.yml)
for saved run artifacts.
