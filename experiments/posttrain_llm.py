"""Zero-shot and fold-safe few-shot five-class sentiment classification.

Default open-weight model: Qwen/Qwen2.5-0.5B-Instruct.
No API key needed. CPU inference can be slow; use CUDA or MPS locally.
No test.csv reads. The 5-shot examples are sampled from OUTER TRAINING FOLDS,
never from validation samples. Invalid JSON outputs are scored as incorrect.
"""
from __future__ import annotations
import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from posttrain_common import (
    load_training_csv, parse_llm_label, prompt_for, save_evaluation,
    splits, training_fewshots, set_seed,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = ("You are a precise movie-review sentiment classifier. "
          "Follow the user's five-class rating instructions. "
          "Return only JSON with the sentiment integer. No explanation.")


def chat_text(tokenizer, review, examples):
    instruction = prompt_for(review, examples)
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": instruction}],
        tokenize=False, add_generation_prompt=True,
    )


def generate_answers(model, tokenizer, prompt_texts, *, batch_size=4,
                     max_input_length=448, max_new_tokens=24, device="cpu"):
    answers = []
    tokenizer.padding_side = "left"
    for start in range(0, len(prompt_texts), batch_size):
        chunk = prompt_texts[start:start + batch_size]
        batch = tokenizer(
            chunk, return_tensors="pt", padding=True,
            truncation=True, max_length=max_input_length,
        )
        input_width = int(batch["input_ids"].shape[1])
        inputs = {key: value.to(device) for key,value in batch.items()}
        with torch.inference_mode():
            sequences = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        suffix = sequences[:, input_width:]
        answers.extend(tokenizer.batch_decode(suffix, skip_special_tokens=True))
        if (start // batch_size + 1) % 5 == 0:
            print(f"LLM generated {min(start + batch_size,len(prompt_texts))}"
                  f"/{len(prompt_texts)}", flush=True)
    return answers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["zero", "few"], required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--train", default=None)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--limit-folds", type=int, default=5)
    p.add_argument("--max-examples-per-fold", type=int, default=0,
                   help="0 means evaluate every validation example")
    p.add_argument("--examples-per-class", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-input-length", type=int, default=448)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="experiments/results/posttrain_llm")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    set_seed(args.seed)
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    frame, y = load_training_csv(args.train)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32
    ).to(args.device).eval()

    pred = np.full(len(y), -1, dtype=int)
    assignments = np.full(len(y), -1, dtype=int)
    invalid_outputs = []
    for fold, (train_idx, valid_idx) in enumerate(splits(y, args.folds, args.seed)):
        if fold >= args.limit_folds:
            break
        rng = np.random.default_rng(args.seed + fold)
        chosen = np.asarray(valid_idx)
        if args.max_examples_per_fold:
            chosen = rng.choice(
                chosen,
                size=min(len(chosen), args.max_examples_per_fold),
                replace=False,
            )
        shots = (training_fewshots(
                    frame, y, train_idx, seed=args.seed+fold, per_class=args.examples_per_class
                 ) if args.mode == "few" else [])
        shot_indices = {i for _, _, i in shots}
        if shot_indices.intersection(set(valid_idx)):
            raise AssertionError("Few-shot leakage: example from validation fold")
        prompts = [chat_text(tokenizer, frame.Phrase.iloc[int(i)], shots) for i in chosen]
        raw = generate_answers(
            model, tokenizer, prompts, batch_size=args.batch_size,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens, device=args.device
        )
        p_fold = np.array([parse_llm_label(reply) for reply in raw], dtype=int)
        pred[chosen] = p_fold
        assignments[chosen] = fold
        invalid_outputs.extend({
            "PhraseId": int(frame.PhraseId.iloc[int(i)]),
            "outer_fold": fold, "raw_response": reply
        } for i, reply in zip(chosen, raw) if parse_llm_label(reply) == -1)
        print(f"LLM mode={args.mode} fold={fold} "
              f"n={len(chosen)} invalid={int((p_fold==-1).sum())}", flush=True)

    output = __import__("pathlib").Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    __import__("json").dump(invalid_outputs,
        (output/"invalid_responses.json").open("w"), indent=2)
    save_evaluation(frame, y, pred, assignments, output, vars(args))


if __name__ == "__main__":
    main()
