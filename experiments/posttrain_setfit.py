"""Train-only SetFit contrastive fine-tuning benchmark for SST-5 movie reviews.

Requires: setfit, sentence-transformers, datasets, torch, sklearn.
Only the outer training fold is passed to SetFit Trainer.
For a smoke check use --limit-folds 1; a score from one fold is NOT OOF.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
import numpy as np
import pandas as pd
import torch
from posttrain_common import load_training_csv, splits, save_evaluation, set_seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default=None)
    p.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--limit-folds", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="experiments/results/posttrain_setfit")
    args = p.parse_args()

    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    torch.set_num_threads(min(torch.get_num_threads(), 3))
    frame, y = load_training_csv(args.train)
    pred = np.full(len(y), -1, dtype=np.int64)
    fold_ids = np.full(len(y), -1, dtype=np.int64)
    for fold, (train_idx, val_idx) in enumerate(splits(y, args.folds, args.seed)):
        if fold >= args.limit_folds:
            break
        set_seed(args.seed + fold)
        # Labels and pair sampling are local to this fold, not whole dataset.
        data = Dataset.from_dict({
            "text": frame.Phrase.iloc[train_idx].astype(str).tolist(),
            "label": y[train_idx].astype(int).tolist(),
        })
        model = SetFitModel.from_pretrained(args.model)
        train_args = TrainingArguments(
            batch_size=args.batch_size,
            num_epochs=1,
            max_steps=args.max_steps,
            samples_per_label=2,
            seed=args.seed + fold,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            show_progress_bar=False,
        )
        trainer = Trainer(model=model, args=train_args, train_dataset=data)
        trainer.train()
        guessed = np.asarray(model.predict(
            frame.Phrase.iloc[val_idx].astype(str).tolist()
        )).astype(int)
        if not np.isin(guessed, np.arange(5)).all():
            raise ValueError("SetFit predicted classes outside 0..4")
        pred[val_idx] = guessed
        fold_ids[val_idx] = fold
        from sklearn.metrics import f1_score, accuracy_score
        print(f"SETFIT fold={fold} macroF1={f1_score(y[val_idx],guessed,average='macro',zero_division=0):.4f} "
              f"accuracy={accuracy_score(y[val_idx],guessed):.4f}", flush=True)
        del model, trainer
    save_evaluation(frame, y, pred, fold_ids, args.output, vars(args))


if __name__ == "__main__":
    main()
