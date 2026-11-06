"""LoRA supervised fine-tuning (SFT) for movie-review JSON sentiment ratings.

The base is an open-weight instruct LLM, not OpenAI's hosted API.
This method fine-tunes only on local training-fold labels, masks prompt tokens
from causal LM loss, and reports validation predictions (no test.csv).

CPU smoke: --limit-folds 1 --max-updates-per-fold 1 --max-valid-per-fold 8
Full OOF: --limit-folds 5 --max-updates-per-fold 0 --max-valid-per-fold 0
For full training, CUDA/MPS is strongly recommended.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from posttrain_common import (
    load_training_csv, prompt_for, parse_llm_label, save_evaluation,
    set_seed, splits,
)
from posttrain_llm import DEFAULT_MODEL, SYSTEM, chat_text, generate_answers


def encode_training_example(tokenizer, review, label, max_length):
    """Create causal LM example with -100 targets on prompt tokens."""
    question = chat_text(tokenizer, review, [])
    prefix = tokenizer.encode(question, add_special_tokens=False)
    completion = tokenizer.encode(
        '{"sentiment": ' + str(int(label)) + "}",
        add_special_tokens=False,
    ) + [tokenizer.eos_token_id]
    if len(completion) >= max_length:
        raise ValueError("Max length too small to fit target JSON")
    prefix = prefix[-(max_length - len(completion)):]
    tokens = prefix + completion
    labels = [-100] * len(prefix) + completion
    return torch.as_tensor(tokens, dtype=torch.long), torch.as_tensor(labels, dtype=torch.long)


def pack(batch, pad_id):
    ids = pad_sequence([example[0] for example in batch], batch_first=True, padding_value=pad_id)
    target = pad_sequence([example[1] for example in batch], batch_first=True, padding_value=-100)
    lengths = torch.as_tensor([len(example[0]) for example in batch])
    attention_mask = (torch.arange(ids.shape[1])[None, :] < lengths[:, None]).long()
    return {"input_ids": ids, "labels": target, "attention_mask": attention_mask}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--train", default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--limit-folds", type=int, default=5)
    p.add_argument("--max-valid-per-fold", type=int, default=0)
    p.add_argument("--max-updates-per-fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="experiments/results/posttrain_llm_sft")
    args = p.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 3))
    frame, y = load_training_csv(args.train)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    p_all = np.full(len(y), -1, dtype=int)
    fold_assign = np.full(len(y), -1, dtype=int)
    for fold, (tr, va) in enumerate(splits(y, args.folds, args.seed)):
        if fold >= args.limit_folds:
            break
        set_seed(args.seed + fold)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32
        )
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank, lora_alpha=args.rank * 2,
            lora_dropout=.05, target_modules=["q_proj", "v_proj"],
        )).to(args.device)
        params = [v for v in model.parameters() if v.requires_grad]
        if not params:
            raise ValueError("No LoRA parameters trainable")
        optimizer = torch.optim.AdamW(params, lr=args.lr)
        rng = np.random.default_rng(args.seed + fold)
        updates = 0
        model.train()
        for epoch in range(args.epochs):
            order = rng.permutation(tr)
            for start in range(0, len(order), args.batch_size):
                selected = order[start:start+args.batch_size]
                batch = [encode_training_example(
                    tokenizer, frame.Phrase.iloc[int(idx)], y[int(idx)], args.max_length
                ) for idx in selected]
                features = pack(batch, tokenizer.pad_token_id)
                features = {key: value.to(args.device) for key, value in features.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = model(**features).loss
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite SFT loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step()
                updates += 1
                print(f"SFT fold={fold} updates={updates} loss={loss.item():.4f}", flush=True)
                if args.max_updates_per_fold and updates >= args.max_updates_per_fold:
                    break
            if args.max_updates_per_fold and updates >= args.max_updates_per_fold:
                break

        model.eval()
        selected_valid = va
        if args.max_valid_per_fold:
            selected_valid = np.random.default_rng(args.seed + fold).choice(
                va, size=min(args.max_valid_per_fold, len(va)), replace=False)
        prompts = [chat_text(tokenizer, frame.Phrase.iloc[int(idx)], [])
                   for idx in selected_valid]
        outputs = generate_answers(
            model, tokenizer, prompts, batch_size=args.batch_size,
            max_input_length=args.max_length, max_new_tokens=24, device=args.device)
        predictions = np.asarray([parse_llm_label(out) for out in outputs], dtype=int)
        p_all[selected_valid] = predictions
        fold_assign[selected_valid] = fold
        print(f"SFT fold={fold} evaluated={len(selected_valid)} "
              f"invalid={(predictions==-1).sum()}", flush=True)
        del model, optimizer
        if args.device == "cuda":
            torch.cuda.empty_cache()
    save_evaluation(frame, y, p_all, fold_assign, args.output, {
        **vars(args), "warning": "Partial folds/updates are an SFT smoke test, not independent full-CV evidence."})


if __name__ == "__main__":
    main()
