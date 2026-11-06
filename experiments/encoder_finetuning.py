"""Local, leakage-controlled sentiment encoder ablations.

Train-only outer 5-fold scores use a separate inner holdout to choose epochs.
The final command refits the selected configuration for the median chosen epoch.
All Hugging Face reads are local-only. No API, downloads or sentence-transformers
installation is needed: the model-card pooling is implemented with Transformers.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, log_loss, mean_absolute_error,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/results/encoder"
CACHE = Path.home() / ".cache/huggingface/hub"
MODEL_INFO = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "mean"),
    "bge_small": ("BAAI/bge-small-en-v1.5", "cls"),
}
# Each auxiliary loss is a matched ablation against the same CE configuration.
CONFIGS = {
    "minilm_ce": dict(model="minilm", last_layers=6),
    "minilm_ce_low_lr": dict(model="minilm", last_layers=6, lr=5e-6),
    "minilm_last2_ce": dict(model="minilm", last_layers=2),
    "minilm_supcon": dict(model="minilm", last_layers=6, supcon=.1),
    "minilm_teacher": dict(model="minilm", last_layers=6, teacher=.2),
    "minilm_ordinal": dict(model="minilm", last_layers=6, ordinal=.1),
    "minilm_hard_negative": dict(model="minilm", last_layers=6, hard_negative=.1),
    "minilm_combined": dict(model="minilm", last_layers=6, supcon=.1,
                            teacher=.2, ordinal=.1, hard_negative=.1),
    "bge_ce": dict(model="bge_small", last_layers=12),
}
LABELS = list(range(5))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.item()
                               if isinstance(x, np.generic) else str(x)))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")


def snapshot(key):
    mid = MODEL_INFO[key][0]
    folder = CACHE / ("models--" + mid.replace("/", "--"))
    choices = sorted(p for p in (folder / "snapshots").glob("*")
                     if (p / "model.safetensors").exists())
    if not choices:
        raise FileNotFoundError(f"No cached checkpoint: {mid}")
    return choices[0]


def metric(y, probs):
    pred = probs.argmax(1)
    report = classification_report(y, pred, labels=LABELS, output_dict=True,
                                   zero_division=0)
    return dict(accuracy=accuracy_score(y, pred), macro_f1=f1_score(
        y, pred, labels=LABELS, average="macro", zero_division=0),
        class0_recall=report["0"]["recall"], nll=log_loss(y, probs, labels=LABELS),
        mae=mean_absolute_error(y, pred), quadratic_kappa=cohen_kappa_score(
            y, pred, labels=LABELS, weights="quadratic"),
        confusion_matrix=confusion_matrix(y, pred, labels=LABELS).tolist(),
        classification_report=report)


class SentimentEncoder(nn.Module):
    def __init__(self, key, last_layers, path=None):
        super().__init__()
        self.key = key
        self.pooling = MODEL_INFO[key][1]
        self.encoder = AutoModel.from_pretrained(path or snapshot(key),
                                                local_files_only=True,
                                                attn_implementation="eager")
        dim = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(.2)
        self.classifier = nn.Linear(dim, 5)
        self.projector = nn.Linear(dim, 16)
        # Fixed class anchors provide a common coordinate system for the teacher.
        self.register_buffer("anchors", torch.eye(16)[:5])
        generator = torch.Generator().manual_seed(17)
        self.register_buffer("semantic_projection", torch.randn(
            dim, 16, generator=generator) / np.sqrt(dim))
        layers = self.encoder.encoder.layer
        if last_layers < len(layers):
            for p in self.encoder.parameters():
                p.requires_grad = False
            for layer in layers[-last_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

    def forward(self, inputs):
        h = self.encoder(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = h[:, 0] if self.pooling == "cls" else (
            h * mask).sum(1) / mask.sum(1).clamp_min(1)
        embedding = F.normalize(pooled, p=2, dim=1)
        # A fixed scale avoids vanishingly small logits from unit length vectors.
        logits = self.classifier(self.dropout(embedding) * 10.)
        projection = F.normalize(self.projector(embedding), p=2, dim=1)
        return logits, embedding, projection


def supervised_contrastive(z, labels, temperature=.1):
    sim = z @ z.T / temperature
    diagonal = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    valid = positives.sum(1) > 0
    if not valid.any():
        return z.sum() * 0
    sim = sim.masked_fill(diagonal, -1e4)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(log_prob * positives).sum(1)[valid].div(
        positives.sum(1)[valid]).mean()


def ordinal_hard_negative(z, labels):
    sim = z @ z.T
    diagonal = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    negative = ~labels[:, None].eq(labels[None, :])
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return z.sum() * 0
    pos = sim.masked_fill(~positive, 1e4).min(1).values
    neg, indices = sim.masked_fill(~negative, -1e4).max(1)
    # Adjacent sentiment classes receive a smaller required separation.
    margin = .05 + .05 * (labels - labels[indices]).abs().float()
    return F.relu(margin + neg[valid] - pos[valid]).mean()


def loss_parts(model, logits, embeddings, projected, labels, teacher_features, cfg):
    ce = F.cross_entropy(logits, labels)
    loss = ce
    parts = {"ce": float(ce.detach())}
    if cfg.get("supcon", 0):
        con = supervised_contrastive(embeddings, labels)
        loss = loss + cfg["supcon"] * con
        parts["supcon"] = float(con.detach())
    if cfg.get("teacher", 0):
        with torch.no_grad():
            target = F.normalize(model.anchors[labels] + .15 * (
                teacher_features @ model.semantic_projection), dim=1)
        dist = (1 - (projected * target).sum(1)).mean()
        loss = loss + cfg["teacher"] * dist
        parts["teacher"] = float(dist.detach())
    if cfg.get("ordinal", 0):
        expected = (logits.softmax(1) * torch.arange(5, device=logits.device)).sum(1)
        ordinal = F.smooth_l1_loss(expected, labels.float())
        loss = loss + cfg["ordinal"] * ordinal
        parts["ordinal"] = float(ordinal.detach())
    if cfg.get("hard_negative", 0):
        hard = ordinal_hard_negative(embeddings, labels)
        loss = loss + cfg["hard_negative"] * hard
        parts["hard_negative"] = float(hard.detach())
    return loss, parts


def tokenize_all(frame, key, max_length=128):
    tok = AutoTokenizer.from_pretrained(snapshot(key), local_files_only=True)
    encoded = tok(frame.Phrase.astype(str).tolist(), truncation=True,
                  padding=True, max_length=max_length, return_tensors="pt")
    return tok, encoded


@torch.inference_mode()
def predict(model, encoded, rows, device, batch_size=32):
    model.eval()
    probabilities, embeddings, projected = [], [], []
    for start in range(0, len(rows), batch_size):
        indices = rows[start:start + batch_size]
        logits, z, p = model({k: v[indices].to(device) for k, v in encoded.items()})
        probabilities.append(logits.softmax(1).cpu().numpy())
        embeddings.append(z.cpu().numpy())
        projected.append(p.cpu().numpy())
    return tuple(np.concatenate(x) for x in (probabilities, embeddings, projected))


def train_run(cfg, encoded, y, frozen, train_rows, val_rows, device, seed,
              epochs=10, fixed_epochs=None, batch_size=24, lr=2e-5):
    seed_all(seed)
    lr = cfg.get("lr", lr)
    model = SentimentEncoder(cfg["model"], cfg["last_layers"]).to(device)
    # Parameters without weight decay follow the standard AdamW BERT convention.
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    # The randomly initialized head must learn faster than the pretrained encoder.
    groups = []
    for is_encoder in [True, False]:
        for decay in [True, False]:
            parameters = [p for n, p in trainable if
                          n.startswith("encoder.") == is_encoder and
                          (not (n.endswith("bias") or "LayerNorm" in n)) == decay]
            if parameters:
                rate = lr if is_encoder else 1e-3
                groups.append(dict(params=parameters, lr=rate, initial_lr=rate,
                                   weight_decay=.01 if decay else 0.))
    optimizer = torch.optim.AdamW(groups)
    history = []
    best_score, best_epoch, stale = -float("inf"), 1, 0
    total_epochs = fixed_epochs if fixed_epochs is not None else epochs
    total_steps = max(1, total_epochs * int(np.ceil(len(train_rows) / batch_size)))
    step = 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, total_epochs + 1):
        model.train()
        running = []
        for indices in np.array_split(rng.permutation(train_rows),
                                      int(np.ceil(len(train_rows) / batch_size))):
            inputs = {k: v[indices].to(device) for k, v in encoded.items()}
            labels = torch.as_tensor(y[indices], device=device)
            teacher = torch.as_tensor(frozen[indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, z, projected = model(inputs)
            loss, parts = loss_parts(model, logits, z, projected, labels, teacher, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            step += 1
            # Warm up through one epoch, then hold constant. This preserves the
            # schedule when refitting for an inner-selected shorter epoch count.
            warmup = max(1, int(np.ceil(len(train_rows) / batch_size)))
            factor = min(step / warmup, 1.)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            optimizer.step()
            running.append(float(loss.detach()))
        row = {"epoch": epoch, "train_loss": float(np.mean(running))}
        if val_rows is not None:
            probs, _, _ = predict(model, encoded, val_rows, device)
            scores = metric(y[val_rows], probs)
            row.update(val_macro_f1=scores["macro_f1"], val_nll=scores["nll"])
            # NLL is a smoother stopping criterion than 80-row macro F1.
            score = -scores["nll"]
            if score > best_score + 1e-5:
                best_score, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
        history.append(row)
        print(json.dumps({"epoch": epoch, "seed": seed, **row}), flush=True)
        if fixed_epochs is None and stale >= 3:
            break
    return model, best_epoch if fixed_epochs is None else fixed_epochs, history


def mine_pairs(frame, frozen, train_rows, path):
    # Only known training labels verify differing sentiment classes. These are
    # label-checked candidates, not human-verified semantic contradictions.
    # These are audit examples. The loss mines current-batch hard negatives in
    # the CURRENT encoder space rather than using this frozen-feature list.
    sim = frozen[train_rows] @ frozen[train_rows].T
    labels = frame.Sentiment.to_numpy()[train_rows]
    sim[labels[:, None] == labels[None, :]] = -np.inf
    neg = train_rows[sim.argmax(1)]
    pairs = pd.DataFrame({"anchor_row": train_rows, "negative_row": neg,
                          "anchor_label": labels,
                          "negative_label": frame.Sentiment.to_numpy()[neg],
                          "cosine": sim.max(1),
                          "anchor_phrase": frame.Phrase.to_numpy()[train_rows],
                          "negative_phrase": frame.Phrase.to_numpy()[neg]})
    assert (pairs.anchor_label != pairs.negative_label).all()
    pairs.to_csv(path, index=False)


def run_cv(args):
    frame = pd.read_csv(ROOT / "train.csv")
    y = frame.Sentiment.to_numpy(dtype=np.int64)
    names = args.configs.split(",") if args.configs else list(CONFIGS)
    for name in names:
        cfg = CONFIGS[name]
        dest = OUT / name
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / "summary.json").exists() and not args.overwrite:
            print(f"Skipping completed {name}", flush=True)
            continue
        frozen = np.load(ROOT / f"experiments/features/{cfg['model']}_train.npy")
        _, encoded = tokenize_all(frame, cfg["model"])
        folds = StratifiedKFold(5, shuffle=True, random_state=42)
        oof = np.zeros((len(frame), 5), dtype=np.float32)
        embeddings = np.zeros((len(frame), frozen.shape[1]), dtype=np.float32)
        all_metrics = []
        for fold, (tr, va) in enumerate(folds.split(frame, y)):
            if args.max_folds and fold >= args.max_folds:
                break
            foldfile = dest / f"fold_{fold}.npz"
            metafile = dest / f"fold_{fold}.json"
            if foldfile.exists() and metafile.exists() and not args.overwrite:
                saved = np.load(foldfile)
                oof[va], embeddings[va] = saved["probabilities"], saved["embeddings"]
                all_metrics.append(json.loads(metafile.read_text()))
                continue
            inner_tr, inner_va = train_test_split(tr, test_size=.2,
                                                  stratify=y[tr], random_state=42+fold)
            mine_pairs(frame, frozen, inner_tr, dest / f"fold_{fold}_inner_hard_pairs.csv")
            print(f"START {name} fold={fold} epoch selection", flush=True)
            start = time.monotonic()
            model, selected_epochs, inner_history = train_run(
                cfg, encoded, y, frozen, inner_tr, inner_va, args.device,
                42+fold, epochs=args.epochs)
            del model
            gc.collect()
            if args.device.type == "mps":
                torch.mps.empty_cache()
            print(f"REFIT {name} fold={fold} epochs={selected_epochs}", flush=True)
            model, _, refit_history = train_run(
                cfg, encoded, y, frozen, tr, None, args.device,
                42+fold, epochs=args.epochs, fixed_epochs=selected_epochs)
            probs, z, projected = predict(model, encoded, va, args.device)
            oof[va], embeddings[va] = probs, z
            scores = metric(y[va], probs)
            scores.update(fold=fold, selected_epochs=selected_epochs,
                          seconds=time.monotonic()-start,
                          frozen_silhouette=float(silhouette_score(frozen[va], y[va])),
                          tuned_silhouette=float(silhouette_score(z, y[va])),
                          inner_train_rows=inner_tr.tolist(), inner_val_rows=inner_va.tolist(),
                          outer_train_rows=tr.tolist(), outer_val_rows=va.tolist(),
                          inner_history=inner_history, refit_history=refit_history)
            np.savez_compressed(foldfile, rows=va, probabilities=probs, embeddings=z,
                                projections=projected)
            write_json(metafile, scores)
            all_metrics.append(scores)
            print(f"DONE {name} fold={fold} macro_f1={scores['macro_f1']:.4f}", flush=True)
            del model
            gc.collect()
            if args.device.type == "mps":
                torch.mps.empty_cache()
        if len(all_metrics) == 5:
            frame_out = frame[["PhraseId", "Sentiment"]].copy()
            frame_out["prediction"] = oof.argmax(1)
            for label in LABELS:
                frame_out[f"prob_{label}"] = oof[:, label]
            frame_out.to_csv(dest / "oof_predictions.csv", index=False)
            np.save(dest / "oof_embeddings.npy", embeddings)
            summary = metric(y, oof)
            summary.update(config=cfg, checkpoint=str(snapshot(cfg["model"])),
                           config_name=name, max_length=128, seed=42,
                           fold_macro_f1=[x["macro_f1"] for x in all_metrics],
                           selected_epochs=[x["selected_epochs"] for x in all_metrics],
                           mean_macro_f1=np.mean([x["macro_f1"] for x in all_metrics]),
                           std_macro_f1=np.std([x["macro_f1"] for x in all_metrics], ddof=1),
                           seconds=sum(x["seconds"] for x in all_metrics),
                           protocol="5 outer folds; separate 20% inner holdout selects epochs by NLL. Fixed configurations; picking among configs adds selection optimism; test untouched.")
            write_json(dest / "summary.json", summary)


def final_fit(args):
    name = args.configs
    cfg = CONFIGS[name]
    summary = json.loads((OUT / name / "summary.json").read_text())
    epochs = int(np.median(summary["selected_epochs"]))
    frame = pd.read_csv(ROOT / "train.csv")
    frozen = np.load(ROOT / f"experiments/features/{cfg['model']}_train.npy")
    tok, encoded = tokenize_all(frame, cfg["model"])
    model, _, history = train_run(cfg, encoded, frame.Sentiment.to_numpy(), frozen,
                                   np.arange(len(frame)), None, args.device, 42,
                                   fixed_epochs=epochs)
    dest = OUT / "final_model"
    dest.mkdir(exist_ok=True)
    model.encoder.save_pretrained(dest / "encoder")
    tok.save_pretrained(dest / "encoder")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()
                if not k.startswith("encoder.")}, dest / "heads.pt")
    train_probs, train_z, train_p = predict(model, encoded, np.arange(len(frame)), args.device)
    np.savez_compressed(dest / "train_outputs.npz", probabilities=train_probs,
                        embeddings=train_z, projections=train_p)
    write_json(dest / "metadata.json", dict(config_name=name, config=cfg, epochs=epochs,
                                            checkpoint=str(snapshot(cfg["model"])),
                                            seed=42, max_length=128, history=history,
                                            train_sha256=hashlib.sha256((ROOT / "train.csv").read_bytes()).hexdigest()))


def smoke(args):
    seed_all(42)
    frame = pd.read_csv(ROOT / "train.csv").groupby("Sentiment").head(4).reset_index(drop=True)
    _, encoded = tokenize_all(frame, "minilm")
    cfg = CONFIGS["minilm_combined"]
    model = SentimentEncoder("minilm", 6).to(args.device)
    original = model.encoder.embeddings.word_embeddings.weight[100].detach().cpu().clone()
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    inputs = {k: v.to(args.device) for k, v in encoded.items()}
    labels = torch.as_tensor(frame.Sentiment.to_numpy(), device=args.device)
    start = time.monotonic()
    logits, z, p = model(inputs)
    # Frozen teacher input must not carry gradients into the target.
    loss, parts = loss_parts(model, logits, z, p, labels, z.detach(), cfg)
    loss.backward()
    grad = float(model.encoder.encoder.layer[-1].attention.self.query.weight.grad.norm())
    assert grad > 0 and np.isfinite(grad)
    optim.step()
    out = predict(model, encoded, np.arange(len(frame)), args.device)
    assert np.isfinite(out[0]).all() and np.allclose(out[0].sum(1), 1., atol=1e-5)
    assert np.allclose(np.linalg.norm(out[1], axis=1), 1., atol=1e-5)
    write_json(OUT / "smoke.json", dict(device=str(args.device), seconds=time.monotonic()-start,
                                         loss=float(loss.detach()), loss_parts=parts,
                                         encoder_gradient_norm=grad, probabilities_finite=True,
                                         unit_embeddings=True))
    print((OUT / "smoke.json").read_text(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["smoke", "cv", "final"])
    ap.add_argument("--configs", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.device = device_for(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Device: {args.device}", flush=True)
    {"smoke": smoke, "cv": run_cv, "final": final_fit}[args.command](args)


if __name__ == "__main__":
    main()
