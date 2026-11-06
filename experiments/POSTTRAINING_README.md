# Four-stage post-training comparison (BigdataGP)

**Goal:** Improve five-class (0–4) movie-review sentiment **Macro-F1**
beyond the existing **0.4971 training-only nested-CV** reference. Accuracy
is secondary. The 341-review test labels remain untouched.

This branch builds on `experiment/target-050-macro-f1`; the earlier
reference model is Amazon-review sentiment RoBERTa frozen embeddings + fold-
trained StandardScaler + RBF-SVM (accuracy 0.5010, macro-F1 0.4971).

## Methods (execution order)

1. **RoBERTa LoRA:** `experiments/posttrain_roberta_lora.py`
   loads `Nav772/roberta-amazon-reviews-5star`, applies a rank-8 PEFT LoRA
   adapter to attention query/value modules, trains its five-class
   classification head, and reports training-fold-only five-fold OOF.
2. **SetFit:** `experiments/posttrain_setfit.py`
   uses `BAAI/bge-small-en-v1.5` to contrastively train sentence embeddings
   on outer training folds and then fits its classification head.
3. **LLM zero-/five-shot:** `experiments/posttrain_llm.py`
   uses `Qwen/Qwen2.5-0.5B-Instruct` in deterministic local CPU/MPS/CUDA
   inference. Five-shot selects one labeled training-fold example per class
   and never uses a validation sample as a demonstration. A strict JSON
   parser rejects invalid/missing/multiple `{"sentiment": n}` responses.
4. **LLM LoRA SFT:** `experiments/posttrain_llm_sft.py`
   trains Qwen LoRA on completion-only five-class JSON targets, masks
   instruction/prompt tokens from the causal loss, and validates only on
   outer held-out folds. CPU smoke validates the pipeline; **a full
   5-fold SFT comparison requires substantial GPU time**.

All scripts read `train.csv` only. One-fold and subset tests are labeled
**smoke checks**, not comparable full-dataset Macro-F1 scores. Cross-
validation uses StratifiedKFold(5, shuffle=True, random_state=42).
Per-model family selection against those folds is still exploratory.

**External pretraining:** Amazon RoBERTa was already trained on Amazon
reviews; SetFit uses a pretrained general embedding model; Qwen is a
pretrained instruction LLM. Verify your course permits external pretrained
weights and, separately, any external supervised task data. No external
SST5 labeled rows are injected by any script in this branch.

## Reproduce

From repository root, install:

```bash
python -m pip install "transformers==4.48.3" "peft==0.14.0" "setfit==1.1.3" \
    "datasets>=2.15,<4" torch scikit-learn pandas pytest

python -m pytest -q tests/test_posttraining.py
```

Full five-fold RoBERTa LoRA (CPU works slowly; use `--device mps` on Apple
Silicon or `--device cuda` on NVIDIA):

```bash
python experiments/posttrain_roberta_lora.py \
    --folds 5 --limit-folds 5 --epochs 1 --max-updates-per-fold 0 \
    --batch-size 8 --device mps --output experiments/results/lora_full
```

Five-fold SetFit:

```bash
python experiments/posttrain_setfit.py \
    --folds 5 --limit-folds 5 --max-steps 16 --device mps \
    --output experiments/results/setfit_full
```

Zero-shot and five-shot (full 511-review evaluation):

```bash
python experiments/posttrain_llm.py --mode zero \
    --folds 5 --limit-folds 5 --max-examples-per-fold 0 --device mps \
    --output experiments/results/qwen_zero
python experiments/posttrain_llm.py --mode few \
    --folds 5 --limit-folds 5 --max-examples-per-fold 0 \
    --examples-per-class 1 --device mps \
    --output experiments/results/qwen_few
```

Full five-fold Qwen LoRA SFT (GPU recommended):

```bash
python experiments/posttrain_llm_sft.py \
    --folds 5 --limit-folds 5 --epochs 1 --max-updates-per-fold 0 \
    --max-valid-per-fold 0 --device cuda \
    --output experiments/results/qwen_sft
```

### Evidence and status

- SetFit 16-step-per-fold full 5-fold CPU: 511 OOF, **accuracy 0.4560,
  Macro-F1 0.3587**. [Successful CI run](https://github.com/SunnyMok1999/BigdataGP/actions/runs/42243830006).
- RoBERTa LoRA smoke (4 optimizer steps, 1 fold, 103 reviews):
  **accuracy 0.4272, Macro-F1 0.4002**. NOT a comparable model score.
  [Successful smoke](https://github.com/SunnyMok1999/BigdataGP/actions/runs/42243837474).
- Qwen zero-shot smoke (12 reviews, one fold): accuracy 0.5000,
  Macro-F1 0.3055. Five-shot smoke (same 12 review indices):
  accuracy 0.5833, Macro-F1 0.4267. These are much too small to rank
  methods. [Smoke CI](https://github.com/SunnyMok1999/BigdataGP/actions/runs/42244065734).
- Qwen LoRA SFT smoke (1 optimizer step, 8 reviews): verifies the
  training/inference pipeline, **not model effectiveness**.
  [SFT smoke](https://github.com/SunnyMok1999/BigdataGP/actions/runs/42244355800).
- Full LoRA five-fold [workflow](https://github.com/SunnyMok1999/BigdataGP/actions/runs/42244303358).
- Full zero/few-shot Qwen [workflow](https://github.com/SunnyMok1999/BigdataGP/actions/runs/42244492503).

All scripts export `predictions.csv` and `metrics.json` with actual scored
rows. Do not compare 8–103 sample smoke outcomes as though they were
complete 511-example out-of-fold measurements.

## Limitations and next actions

- SetFit's short 16-step training schedule may be underfitting; report that
  configuration exactly, do not infer SetFit can never help.
- Running many model families on the same five outer folds can lead to
  **selection optimism**. A score slightly above 0.50 would still require
  carefully locked final evaluation.
- Completion-only Qwen SFT is defined but full CV may need a suitable GPU
  host, plus a separate set of experiments for epoch selection.
- Keep the 341-review test untouched until the model/prompt/training
  configuration is locked. An independent test score is not guaranteed.

## Full cross-validation results (completed)

All reported scores use the original 511 `train.csv` rows, five stratified
outer folds and exactly the specified fixed training/prompt configuration.
None of the four methods has yet beaten the 0.4971 Macro-F1 reference.

| Method | Train-only OOF accuracy | Macro-F1 | Evaluation |
|---|---:|---:|---|
| Frozen Amazon RoBERTa + StandardScaler + RBF SVM | **0.5010** | **0.4971** | Prior nested-CV reference |
| Amazon RoBERTa LoRA, 1 full epoch/fold | 0.4521 | 0.4217 | [successful five-fold CI](https://github.com/SunnyMok1999/BigdataGP/actions/runs/42244303358) |
| SetFit BGE-small, 16 training steps/fold | 0.4560 | 0.3587 | [successful five-fold CI](https://github.com/SunnyMok1999/BigdataGP/actions/runs/42243830006) |
| Qwen2.5-0.5B Zero-shot | 0.3875 | 0.3418 | [full zero/few CI](https://github.com/SunnyMok1999/BigdataGP/actions/runs/42244492503) |
| Qwen2.5-0.5B Five-shot, one example per class | 0.3914 | 0.3857 | same full CI |

The Qwen zero-shot produced valid JSON for every row; five-shot produced
one invalid response (counted as wrong). Five-shot improves Macro-F1 by
+0.0440 over zero-shot but is still below the frozen RoBERTa baseline.
Qwen LoRA SFT has only a one-update, eight-review smoke test so far;
its 0.10 smoke Macro-F1 is not a full-data benchmark and is deliberately
excluded from the ranked full-data table.

Additional SetFit 64- and 128-training-step configurations can be inspected
in [the step-budget workflow](https://github.com/SunnyMok1999/BigdataGP/actions/runs/42245041328).
