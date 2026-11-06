"""Extract TRAIN-ONLY off-domain sentiment model logits as transferable features.

Never read test.csv. Product-star model was trained on product reviews,
not SST-5; check provenance before treating any external checkpoint as an
independent benchmark. Logits are fixed features: no labels are used here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "review5": "nlptown/bert-base-multilingual-uncased-sentiment",
    "tweet3": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "yelp5": "AdhamEhab/fine-tuned-bert-yelp",
}


def hash_sentences(frame):
    h = hashlib.sha256()
    for id_, text in zip(frame["PhraseId"], frame["Phrase"]):
        h.update((str(int(id_)) + "\t" + str(text) + "\n").encode("utf-8"))
    return h.hexdigest()


def extract(args):
    frame = pd.read_csv(args.train, usecols=["PhraseId", "Phrase"])
    if not frame.PhraseId.is_unique or frame.Phrase.isna().any():
        raise ValueError("Invalid training rows")
    name = MODELS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(args.device).eval()
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    texts = frame.Phrase.astype(str).tolist()
    outputs = []
    hidden_outputs = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            chunk = texts[start:start + args.batch_size]
            batch = tokenizer(chunk, padding=True, truncation=True,
                              max_length=args.max_length, return_tensors="pt")
            batch = {k: v.to(args.device) for k, v in batch.items()}
            result = model(**batch, output_hidden_states=bool(args.hidden_output))
            outputs.append(result.logits.float().cpu().numpy())
            if args.hidden_output:
                hidden_outputs.append(result.hidden_states[-1][:, 0, :].float().cpu().numpy())
            if (start // args.batch_size + 1) % 10 == 0:
                print(f"{args.model}: extracted {min(start+args.batch_size,len(texts))}/{len(texts)}", flush=True)
    logits = np.concatenate(outputs).astype(np.float32)
    if logits.shape != (len(frame), model.config.num_labels) or not np.isfinite(logits).all():
        raise AssertionError("Malformed model logits")
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, logits)
    frame[["PhraseId"]].to_csv(dest.with_suffix(".rows.csv"), index=False)
    metadata = {
        "model": name, "alias": args.model,
        "model_labels": model.config.id2label,
        "rows": len(frame), "shape": list(logits.shape),
        "phrase_id_and_text_sha256": hash_sentences(frame),
        "train_csv_sha256": hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
        "max_length": args.max_length,
        "note": "Fixed pretrained sentiment logits; train.csv text only; external model not tuned on local labels."
    }
    dest.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    if args.hidden_output:
        vectors = np.concatenate(hidden_outputs).astype(np.float32)
        norms = np.maximum(1e-8, np.linalg.norm(vectors, axis=1, keepdims=True))
        vectors = vectors / norms
        hidden_path = Path(args.hidden_output)
        hidden_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(hidden_path, vectors)
        frame[["PhraseId"]].to_csv(hidden_path.with_name(hidden_path.stem + "_rows.csv"),index=False)
        hidden_path.with_name(hidden_path.stem + "_metadata.json").write_text(
            json.dumps({**metadata,"shape":list(vectors.shape),
                        "feature_type":"last transformer layer CLS, L2-normalized",
                        "source_logits_path":str(dest)},indent=2))
        print("saved sentiment hidden vectors",hidden_path, vectors.shape,flush=True)
    print("saved", str(dest), logits.shape, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--train", default=str(ROOT / "train.csv"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", choices=("cpu","cuda","mps"), default="cpu")
    p.add_argument("--output", required=True)
    p.add_argument("--hidden-output", default=None, help="Save final-layer CLS features for nested CV")
    extract(p.parse_args())


if __name__ == "__main__":
    main()
