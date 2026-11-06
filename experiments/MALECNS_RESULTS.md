# MaleCNS sentiment: training-only experimental results

**Status:** Completed CPU training-only experiments on 6 Nov 2026.
**Source:** BigdataGP `train.csv` (511 examples, 5 labels); untouched
`test.csv` (341 examples).
**Source graph:** Janelia MaleCNS v1.0, reused from the provenance-documented
80-neuron, 1,296-edge circuit in `cobanov/flyjump` (CC BY 4.0).

**Important:** These are exploratory, non-nested graph-family comparisons,
not independent test scores. They do NOT show that biological wiring is
better than matched random wiring or that Macro-F1 above 0.5 was achieved.

## A. Learned word embeddings + sparse recurrent network

5-fold StratifiedKFold (shuffle=True, random_state=42), fixed 12 epochs,
seeds 42/123/2026, square-root inverse-frequency class weights, same folds
and initialization seeds for each topology. Numbers below are the **mean
across three complete OOF Macro-F1 scores**, not the mean-fold CSV values.

| Nodes | MaleCNS | Degree-preserving rewire | Edge-matched random |
|---:|---:|---:|---:|
| 32 | 0.1998 | 0.2030 | 0.1956 |
| 64 | 0.2188 | 0.2197 | 0.2233 |
| 80 | 0.2175 | 0.2232 | 0.2196 |

Mean OOF accuracy across seeds for 32: 0.2257 / 0.2302 / 0.2205;
64: 0.2413 / 0.2433 / 0.2466;
80: 0.2446 / 0.2485 / 0.2446 (same graph column order).

Workflow passed all 4 tests and every scheduled training job:
https://github.com/SunnyMok1999/BigdataGP/actions/runs/42201634173

Workflow artifact `malecns-full-training-only-cv` contains fold results,
out-of-fold predictions, per-class recall, confusion matrices and config.

## B. Frozen BGE-small sentence vectors + recurrent graph

5-fold StratifiedKFold, 80-node graph, fixed 30 epochs, 6 recurrent updates
with constant input, seeds 42/123/2026.

| Topology | Mean OOF accuracy | Mean OOF Macro-F1 |
|---|---:|---:|
| MaleCNS | 0.4077 | 0.3940 |
| Degree-preserving rewire | 0.4109 | 0.3973 |
| Edge-matched random | 0.4149 | 0.4013 |
| Existing BGE-small + LR nested-CV reference | 0.4188 | 0.4148 |

The BGE + LR reference has a **different model-selection and architecture
protocol**. Its score is not a matched topology ablation.

Individual BGE OOF Macro-F1 by seed:

| Seed | MaleCNS | Rewired | Random |
|---:|---:|---:|---:|
| 42 | 0.3853 | 0.3998 | 0.3928 |
| 123 | 0.3796 | 0.3854 | 0.3895 |
| 2026 | 0.4171 | 0.4067 | 0.4217 |

Workflow passed all 6 tests and training:
https://github.com/SunnyMok1999/BigdataGP/actions/runs/42202082028

Workflow artifact `malecns-bge-train-only` contains full OOF predictions
and confusion matrices.

## Interpretation and next steps

1. The biological graph does **not** consistently beat both topology
   controls. No biological-topology inductive-bias gain has been established.
2. Frozen pretrained language features improve classification much more than
   the specific measured subgraph does. Even BGE-driven recurrence did not
   beat the existing BGE + LR reference.
3. Models share folds, but the three training seeds are not independent data
   replications. Selection among graph size/epoch/family still induces optimism.
4. Recommended next comparison: freeze the BGE representation, test a
   parameter-matched *nonrecurrent* MLP and a sparse recurrent graph with
   controlled trainable parameter counts. Consider small training-fold-only
   tuning/nested CV if pursuing an actual performance improvement.
5. Keep `test.csv` completely held out until one method is locked.


## C. Vanilla RNN and LSTM versus MaleCNS (new)

Identical 5-fold stratified partitions (seed 42), fixed **12 epochs** per fold,
training seeds **42, 123, 2026**, hash-tokenized raw text (48D learned
embeddings), hidden size **80**, final-plus-mean-state pooled classification,
class-weighted CE and AdamW. All methods are unidirectional. The measured
graph is rerun in the same workflow to verify the comparison.

**Mean of the three complete 511-review training OOF scores:**

| Architecture | Mean OOF accuracy | Mean OOF Macro-F1 | Stored parameter count |
|---|---:|---:|---:|
| 80-neuron MaleCNS sparse RNN | 0.2446 | 0.2175 | 404,789 (includes masked zeros) |
| Standard dense tanh RNN (80 hidden) | 0.2290 | 0.2090 | 404,789 |
| Standard LSTM (80 hidden) | 0.2479 | 0.2187 | 435,989 |

**Per-seed OOF Macro-F1:**

| Seed | MaleCNS | Dense RNN | LSTM |
|---:|---:|---:|---:|
| 42 | 0.2002 | 0.2001 | 0.2121 |
| 123 | 0.2367 | 0.2131 | 0.2078 |
| 2026 | 0.2157 | 0.2138 | 0.2363 |

Workflow: https://github.com/SunnyMok1999/BigdataGP/actions/runs/42202978266

The workflow successfully passed **8 tests** covering graph invariants,
padding, finite gradients, and held-out-label invariance; uploaded complete
fold metrics, per-seed metrics, probabilities and configurations as the
`rnn-lstm-malecns-training-only-comparison` artifact.

The LSTM leads by just **+0.0012** Macro-F1 over MaleCNS when averaging
three seeds; it is not an established improvement. The measured graph does
slightly better than vanilla RNN on these runs, but different effective
trainable recurrent connectivity and limited training data make attribution
to biological topology unwarranted. The ordinary RNN and masked MaleCNS
matrices have the same *allocated* number of weights; masked graph zeros
receive zero gradients and are not functionally trainable. The gated
LSTM is a different parameterization. No claim of parameter-matched
inductive-bias advantage or statistical significance is justified.

All three token-only models underperform the existing frozen BGE-small + LR
train-only nested-CV reference (Macro-F1 0.4148), though that reference
uses a different tuning protocol. The comparison reinforces that strong
pretrained language representations matter more for this small dataset
than adding recurrence from scratch.
