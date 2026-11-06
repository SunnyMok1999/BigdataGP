"""Train-only RoBERTa LoRA domain adaptation for five sentiment classes.

Default model was *already* trained for 5-star Amazon review sentiment.
PEFT LoRA learns from ONLY local training-fold labels; test.csv is never read.
Full benchmark: --limit-folds 5 --max-updates-per-fold 0 --epochs 2
CPU smoke benchmark: --limit-folds 1 --max-updates-per-fold 4
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model

from posttrain_common import load_training_csv, splits, save_evaluation, set_seed

MODEL = "Nav772/roberta-amazon-reviews-5star"


def configure_model(model_id: str, rank: int, dropout: float):
    base = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=5)
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank, lora_alpha=2 * rank, lora_dropout=dropout,
        target_modules=["query", "value"], bias="none",
        modules_to_save=["classifier"],
    )
    model = get_peft_model(base, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0 or trainable >= total:
        raise AssertionError("Expected nonzero, parameter-efficient LoRA weights")
    return model, trainable, total


def validate_lengths_and_logits(model, token_batch):
    out = model(**token_batch).logits
    if out.shape != (len(token_batch["input_ids"]), 5):
        raise AssertionError("Expected batch x 5 logits")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--train", default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--limit-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-updates-per-fold", type=int, default=0,
                   help="0=all batches; positive=SMOKE ONLY: cap optimizer updates")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=96)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--adapter-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--output", default="experiments/results/posttrain_lora")
    args = p.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(min(3, torch.get_num_threads()))
    frame, y = load_training_csv(args.train)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokens = tokenizer(
        frame.Phrase.astype(str).tolist(), padding="max_length",
        truncation=True, max_length=args.max_length, return_tensors="pt",
    )
    preds = np.full(len(y), -1, dtype=int)
    fold_ids = np.full(len(y), -1, dtype=int)
    generator = np.random.default_rng(args.seed)
    model_counts = None
    for fold, (train_idx, val_idx) in enumerate(splits(y, args.folds, args.seed)):
        if fold >= args.limit_folds:
            break
        set_seed(args.seed + fold)
        model, trainable, total = configure_model(args.model, args.rank, .1)
        model_counts = {"trainable": trainable, "total": total}
        model = model.to(args.device)
        groups = [
            {"params": [p for n,p in model.named_parameters()
                        if p.requires_grad and "lora_" in n], "lr": args.adapter_lr},
            {"params": [p for n,p in model.named_parameters()
                        if p.requires_grad and "lora_" not in n], "lr": args.head_lr},
        ]
        if not all(group["params"] for group in groups):
            raise AssertionError("LoRA and classifier parameters must both be trainable")
        optim = torch.optim.AdamW(groups, weight_decay=.01)
        counts = np.bincount(y[train_idx], minlength=5).astype(float)
        weights = torch.as_tensor(np.sqrt(len(train_idx) / (5 * np.maximum(counts, 1))),
                                  device=args.device, dtype=torch.float)
        rng = np.random.default_rng(args.seed + fold)
        updates = 0
        model.train()
        for epoch in range(args.epochs):
            order = rng.permutation(train_idx)
            for start in range(0, len(order), args.batch_size):
                chosen = order[start:start+args.batch_size]
                features = {k: v[chosen].to(args.device) for k,v in tokens.items()}
                label = torch.as_tensor(y[chosen], device=args.device)
                optim.zero_grad(set_to_none=True)
                logits = validate_lengths_and_logits(model, features)
                loss = F.cross_entropy(logits, label, weight=weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.
                )
                optim.step()
                updates += 1
                if updates % 10 == 0:
                    print(f"LoRA fold={fold} step={updates} loss={loss.item():.4f}", flush=True)
                if args.max_updates_per_fold and updates >= args.max_updates_per_fold:
                    break
            if args.max_updates_per_fold and updates >= args.max_updates_per_fold:
                break
        model.eval()
        fold_predictions = []
        with torch.inference_mode():
            for start in range(0, len(val_idx), args.batch_size):
                chosen = val_idx[start:start+args.batch_size]
                features = {k: v[chosen].to(args.device) for k,v in tokens.items()}
                fold_predictions.extend(
                    validate_lengths_and_logits(model, features).argmax(-1).cpu().tolist()
                )
        p_fold = np.asarray(fold_predictions, dtype=int)
        preds[val_idx] = p_fold
        fold_ids[val_idx] = fold
        print(f"LORA fold={fold} updates={updates} macroF1="
              f"{f1_score(y[val_idx],p_fold,average='macro',zero_division=0):.4f} "
              f"accuracy={accuracy_score(y[val_idx],p_fold):.4f} "
              f"trainable={trainable}/{total}",flush=True)
        del model, optim
        if args.device == "cuda":
            torch.cuda.empty_cache()
    meta = {**vars(args), "peft_weights": model_counts,
            "warning": "Capped updates and partial folds are smoke results, not a full model comparison."}
    save_evaluation(frame, y, preds, fold_ids, args.output, meta)


if __name__ == "__main__":
    main()
