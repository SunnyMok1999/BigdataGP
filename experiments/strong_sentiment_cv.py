"""Strong small-data sentiment probes with honest nested train-only CV.

All model fitting and supervised hyperparameter selection happen inside
the outer training fold. The 341 test examples are NEVER loaded.
Features: cached pretrained BGE-small and optional off-domain 5-star logits.

Usage:
python experiments/strong_sentiment_cv.py --candidates bge_lr bge_svm bge_linear ...
python experiments/strong_sentiment_cv.py --star experiments/features/review5_train.npy
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "experiments/features"
ALL = ["bge_lr", "bge_svm", "bge_linear", "bge_multiview_lr",
       "transfer_direct", "transfer_only_lr", "bge_transfer_lr", "bge_transfer_svm",
       "sent_hidden_lr", "sent_hidden_svm", "bge_hidden_lr", "bge_hidden_svm",
       "sent_hidden_linear", "sent_hidden_poly", "sent_hidden_scaled_svm"]


class BlockScaler(BaseEstimator, TransformerMixin):
    """Normalize independently input sentence embeddings and transferred logits.

    X starts with 384-dim unit-L2 BGE. Transferred off-domain logits follow.
    Fit StandardScaler ONLY on fold-training transfer-logit rows.
    """
    def __init__(self, width=384, extra_weight=1.0):
        self.width = width
        self.extra_weight = extra_weight

    def fit(self, X, y=None):
        self.extra_scaler_ = StandardScaler().fit(X[:, self.width:])
        return self

    def transform(self, X):
        a = X[:, :self.width]
        b = self.extra_scaler_.transform(X[:, self.width:])
        b = b / np.maximum(1e-8, np.linalg.norm(b, axis=1, keepdims=True))
        return np.hstack((a, b * self.extra_weight))


def load_embedding(path: Path, df: pd.DataFrame):
    X = np.load(path, allow_pickle=False)
    rows = pd.read_csv(path.with_name(path.stem + "_rows.csv"))
    if X.shape[0] != len(df) or not np.array_equal(df.PhraseId, rows.PhraseId):
        raise ValueError(f"Row mismatch: {path}")
    if "Sentiment" in rows and not np.array_equal(df.Sentiment, rows.Sentiment):
        raise ValueError("Label mismatch in embedding row metadata")
    return X.astype(np.float64)


def load_transfer(path: Path, df: pd.DataFrame):
    X = np.load(path, allow_pickle=False)
    row_csv = path.with_suffix(".rows.csv")
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    rows = pd.read_csv(row_csv)
    if X.shape[0] != len(df) or X.shape[1] not in (3,5,8):
        raise ValueError("Expected N x 3, N x 5, or N x 8 transfer logits")
    if not np.array_equal(df.PhraseId, rows.PhraseId):
        raise ValueError("PhraseId order mismatch for transferred logits")
    h = hashlib.sha256()
    for id_, t in zip(df.PhraseId, df.Phrase):
        h.update((str(int(id_)) + "\t" + str(t) + "\n").encode())
    if metadata["phrase_id_and_text_sha256"] != h.hexdigest():
        raise ValueError("Text mismatch for transferred logits")
    return X.astype(np.float64), metadata


def candidate_spec(name: str, bge: np.ndarray, multi: np.ndarray | None,
                   extra: np.ndarray | None, hidden: np.ndarray | None = None):
    if name == "bge_lr":
        estimator = Pipeline([("clf", LogisticRegression(max_iter=2500))])
        grid = {"clf__C": [.01,.1,1.,10.,100.], "clf__class_weight":[None,"balanced"]}
        X = bge
    elif name == "bge_svm":
        estimator = Pipeline([("clf", SVC(kernel="rbf"))])
        grid = {"clf__C": [.1,1.,10.,100.],
                "clf__gamma": [.1,.5,1.,2.],
                "clf__class_weight":[None,"balanced"]}
        X = bge
    elif name == "bge_linear":
        estimator = Pipeline([("clf", SVC(kernel="linear"))])
        grid = {"clf__C":[.01,.1,1.,10.,100.],
                "clf__class_weight":[None,"balanced"]}
        X = bge
    elif name == "bge_multiview_lr":
        if multi is None:
            raise ValueError("Multi-view features unavailable")
        estimator = Pipeline([("clf", LogisticRegression(max_iter=2500))])
        grid = {"clf__C":[.01,.1,1.,10.,100.],
                "clf__class_weight":[None,"balanced"]}
        X = multi
    elif name == "transfer_direct":
        if extra is None or extra.shape[1] != 5:
            raise ValueError("Direct transfer requires 5-class logits")
        return extra, None, None
    elif name == "transfer_only_lr":
        if extra is None:
            raise ValueError("Missing transfer features")
        estimator = Pipeline([("scale",StandardScaler()),("clf",LogisticRegression(max_iter=2500))])
        grid = {"clf__C":[.01,.1,1.,10.,100.],
                "clf__class_weight":[None,"balanced"]}
        X = extra
    elif name == "bge_transfer_lr":
        if extra is None:
            raise ValueError("Missing transfer features")
        X = np.hstack((bge,extra))
        estimator = Pipeline([("blocks",BlockScaler(width=bge.shape[1])),
                              ("clf",LogisticRegression(max_iter=2500))])
        grid = {"blocks__extra_weight":[.25,.5,1.,2.,4.],
                "clf__C":[.01,.1,10.,100.],
                "clf__class_weight":[None,"balanced"]}
    elif name == "bge_transfer_svm":
        if extra is None:
            raise ValueError("Missing transfer features")
        X = np.hstack((bge,extra))
        estimator = Pipeline([("blocks",BlockScaler(width=bge.shape[1])),
                              ("clf",SVC(kernel="rbf"))])
        grid = {"blocks__extra_weight":[.25,1.,2.],
                "clf__C":[.1,1.,10.,100.],
                "clf__gamma":[.1,.5,1.],
                "clf__class_weight":[None,"balanced"]}
    elif name in ("sent_hidden_lr", "sent_hidden_svm", "bge_hidden_lr", "bge_hidden_svm"):
        if hidden is None:
            raise ValueError("Missing external sentiment-tuned hidden features")
        combined = name.startswith("bge_")
        X = np.hstack((bge,hidden)) if combined else hidden
        if "lr" in name:
            clf = LogisticRegression(max_iter=2500)
            search_grid = {"clf__C":[.01,.1,1.,10.,100.],
                           "clf__class_weight":[None,"balanced"]}
        else:
            clf = SVC(kernel="rbf")
            search_grid = {"clf__C":[.1,1.,10.,100.],
                           "clf__gamma":[.1,.5,1.,2.,4.],
                           "clf__class_weight":[None,"balanced"]}
        if combined:
            estimator=Pipeline([("blocks",BlockScaler(width=bge.shape[1])),("clf",clf)])
            grid={"blocks__extra_weight":[.25,.5,1.,2.,4.],**search_grid}
        else:
            estimator=Pipeline([("clf",clf)])
            grid=search_grid
    elif name in ("sent_hidden_linear", "sent_hidden_poly", "sent_hidden_scaled_svm"):
        if hidden is None:
            raise ValueError("Missing external sentiment-tuned hidden features")
        X = hidden
        if name == "sent_hidden_linear":
            estimator = Pipeline([("clf",SVC(kernel="linear"))])
            grid = {"clf__C":[.001,.01,.1,1.,10.,100.],
                    "clf__class_weight":[None,"balanced"]}
        elif name == "sent_hidden_poly":
            estimator = Pipeline([("clf",SVC(kernel="poly",degree=2,coef0=1))])
            grid = {"clf__C":[.1,1.,10.,100.],
                    "clf__gamma":[.1,.5,1.,2.],
                    "clf__class_weight":[None,"balanced"]}
        else:
            estimator = Pipeline([("scale",StandardScaler()),("clf",SVC(kernel="rbf"))])
            grid = {"clf__C":[.1,1.,10.,100.],
                    "clf__gamma":["scale",.0001,.001,.01],
                    "clf__class_weight":[None,"balanced"]}
    else:
        raise ValueError(f"Unknown candidate {name}")
    return X, estimator, grid


def benchmark(args):
    df = pd.read_csv(args.train)
    if not {"PhraseId","Phrase","Sentiment"}.issubset(df) or not df.PhraseId.is_unique:
        raise ValueError("train.csv requires unique PhraseId, Phrase and Sentiment")
    y = df.Sentiment.to_numpy(dtype=np.int64)
    if set(y) != set(range(5)):
        raise ValueError("Expected five labels 0..4")
    bge = load_embedding(F/"bge_small_train.npy",df)
    multi = None
    if "bge_multiview_lr" in args.candidates:
        d = load_embedding(F/"distilbert_train.npy",df)
        bt = load_embedding(F/"bert_base_train.npy",df)
        multi = np.hstack((bge, d, bt))
    transfer, provenance = (load_transfer(Path(args.transfer),df) if args.transfer else (None,None))
    hidden = load_embedding(Path(args.sentiment_embeddings),df) if args.sentiment_embeddings else None
    outer = list(StratifiedKFold(5,shuffle=True,random_state=42).split(bge,y))
    result = []
    oof_rows = []
    fold_records = []
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    for name in args.candidates:
        X, classifier, grid = candidate_spec(name,bge,multi,transfer,hidden)
        pred = np.full(len(y), -1,dtype=int)
        folds = np.full(len(y),-1,dtype=int)
        for fold,(train_ids, val_ids) in enumerate(outer):
            if classifier is None:
                this_pred = softmax(X[val_ids],axis=1).argmax(axis=1)
                best_params = {"fixed_pretrained":True}
                inner_score = None
            else:
                search = GridSearchCV(
                    classifier,grid,scoring="f1_macro",refit=True,n_jobs=1,
                    cv=StratifiedKFold(3,shuffle=True,random_state=42+fold),
                    error_score="raise",
                )
                search.fit(X[train_ids],y[train_ids])
                this_pred = search.predict(X[val_ids])
                best_params = search.best_params_
                inner_score = float(search.best_score_)
            pred[val_ids] = this_pred
            folds[val_ids] = fold
            fold_metrics = {
                "candidate":name,"fold":fold,"macro_f1":f1_score(y[val_ids],this_pred,average="macro",zero_division=0),
                "accuracy":accuracy_score(y[val_ids],this_pred),
                "inner_best_f1":inner_score,"best_params":str(best_params)
            }
            fold_records.append(fold_metrics)
            print(f"{name} fold {fold}: F1={fold_metrics['macro_f1']:.4f}, ACC={fold_metrics['accuracy']:.4f}",flush=True)
        assert (folds>=0).all() and (pred>=0).all()
        row={"candidate":name,"accuracy":accuracy_score(y,pred),
             "macro_f1":f1_score(y,pred,average="macro",zero_division=0),
             "confusion_matrix":confusion_matrix(y,pred,labels=np.arange(5)).tolist(),
             "classification_report":classification_report(y,pred,labels=np.arange(5),output_dict=True,zero_division=0)}
        result.append(row)
        for i in range(len(y)):
            oof_rows.append({"candidate":name,"row_index":i,"PhraseId":int(df.PhraseId.iloc[i]),
                             "outer_fold":folds[i],"prediction":int(pred[i])})
        print(f"OOF {name}: macroF1={row['macro_f1']:.4f}, accuracy={row['accuracy']:.4f}",flush=True)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ("classification_report","confusion_matrix")} for r in result]).to_csv(out/"summary.csv",index=False)
    pd.DataFrame(fold_records).to_csv(out/"fold_metrics.csv",index=False)
    pd.DataFrame(oof_rows).to_csv(out/"oof_predictions.csv",index=False)
    (out/"details.json").write_text(json.dumps({
        "protocol":"train-only 5 outer / 3 inner stratified CV; separate hyperparameter tuning inside each outer fold",
        "models":result,"transfer_provenance":provenance,
        "warning":"Selecting the best among candidate families based on these OOF results still adds selection optimism; no test.csv used.",
    },indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",default=str(ROOT/"train.csv"))
    p.add_argument("--transfer",default=None,help="Optional precomputed train-only off-domain sentiment logits")
    p.add_argument("--sentiment-embeddings",default=None,help="Optional off-domain pretrained CLS vector cache; requires _rows.csv")
    p.add_argument("--candidates",nargs="+",choices=ALL,default=["bge_lr","bge_svm","bge_linear","bge_multiview_lr"])
    p.add_argument("--output",default=str(ROOT/"experiments/results/strong_cv"))
    args=p.parse_args()
    benchmark(args)


if __name__=="__main__":
    main()
