"""Leakage-audited extra-data experiment for five-class movie sentiment.

Uses public SetFit/sst5 TRAIN labels for extra supervision. The local
BigDataGP test is NEVER scored: only its TEXT is read to exclude potential
external dataset duplicates. This external-data track must not be compared
as a 511-label-only benchmark or submitted if course rules prohibit it.

No local held-out-fold labels enter external filtering or model fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from rapidfuzz import fuzz, process
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer

from strong_sentiment_cv import ROOT, F, load_embedding


def normalize(text):
    s=str(text).lower().replace("-lrb-", "(").replace("-rrb-", ")")
    return " ".join(re.findall(r"[a-z0-9]+",s))


def prepare_external(train, test_text_only, threshold):
    data=load_dataset("SetFit/sst5",split="train")
    if "text" not in data.column_names or "label" not in data.column_names:
        raise ValueError("Unexpected external data schema")
    local_keys={normalize(x) for x in train["Phrase"]}
    test_keys={normalize(x) for x in test_text_only["Phrase"]}
    forbidden=sorted(local_keys | test_keys)
    remaining=[]
    exact=near=0
    seen=set()
    for row in data:
        norm=normalize(row["text"])
        if not norm or norm in seen:
            continue
        seen.add(norm)
        if norm in local_keys or norm in test_keys:
            exact+=1
            continue
        # Compare every public sentence with all local TRAIN and TEST texts.
        # Content-only decontamination, not test-label selection.
        hit=process.extractOne(norm,forbidden,scorer=fuzz.ratio,score_cutoff=threshold)
        if hit is not None:
            near+=1
            continue
        remaining.append((str(row["text"]),int(row["label"])))
    if len(remaining)<2000:
        raise ValueError("Too few independent external rows after filtering")
    y=np.array([i[1] for i in remaining],dtype=np.int64)
    if set(y)!=set(range(5)):
        raise ValueError("External labels are not five ordered classes")
    return [i[0] for i in remaining],y,{"dataset":"SetFit/sst5","split":"train",
        "total_external_rows":len(data),"external_after_filter":len(remaining),
        "exact_overlap_removed":exact,"near_overlap_removed":near,
        "threshold_fuzz_ratio":threshold,
        "local_train_text_rows":len(train),"local_test_text_rows":len(test_text_only),
        "disclaimer":"Test TEXT was read for decontamination; test labels NEVER accessed."}


def encode(texts,args):
    torch.set_num_threads(4)
    model_id="BAAI/bge-small-en-v1.5"
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    model=AutoModel.from_pretrained(model_id).eval()
    vectors=[]
    with torch.inference_mode():
        for begin in range(0,len(texts),args.batch_size):
            batch=tokenizer(texts[begin:begin+args.batch_size],
                padding=True,truncation=True,max_length=128,return_tensors="pt")
            z=model(**batch).last_hidden_state[:,0,:].float().numpy()
            z=z/np.maximum(1e-8,np.linalg.norm(z,axis=1,keepdims=True))
            vectors.append(z.astype(np.float32))
            if begin//args.batch_size%30==0:
                print("Encoded external",min(begin+args.batch_size,len(texts)),"/",len(texts),flush=True)
    return np.concatenate(vectors)


def cv(args,Xext,yext,Xlocal,ylocal,out,metadata):
    outer=list(StratifiedKFold(5,shuffle=True,random_state=42).split(Xlocal,ylocal))
    pred=np.full(len(ylocal),-1,dtype=int)
    fold_records=[]
    # Zero-shot external-only training is valid: external sentences were
    # de-duplicated against *every* local train and test text in advance.
    base=SVC(kernel="rbf",C=10,gamma=1,class_weight="balanced")
    base.fit(Xext,yext)
    direct=base.predict(Xlocal)
    print("EXTERNAL-ONLY macroF1",round(f1_score(ylocal,direct,average="macro",zero_division=0),4),
          "accuracy",round(accuracy_score(ylocal,direct),4),flush=True)

    for fold,(tr,va) in enumerate(outer):
        inners=list(StratifiedKFold(3,shuffle=True,random_state=42+fold).split(Xlocal[tr],ylocal[tr]))
        candidates=[]
        for c in (1.,10.):
            for g in (.5,1.):
                for balance in (None,"balanced"):
                    fold_scores=[]
                    for it,iv in inners:
                        inner_train=tr[it]
                        inner_val=tr[iv]
                        model=SVC(kernel="rbf",C=c,gamma=g,class_weight=balance)
                        model.fit(np.vstack([Xext,Xlocal[inner_train]]),
                                  np.concatenate([yext,ylocal[inner_train]]))
                        p=model.predict(Xlocal[inner_val])
                        fold_scores.append(f1_score(ylocal[inner_val],p,average="macro",zero_division=0))
                    candidates.append((float(np.mean(fold_scores)),c,g,balance))
        candidates.sort(key=lambda item:(-item[0],item[1],item[2],str(item[3])))
        best=candidates[0]
        model=SVC(kernel="rbf",C=best[1],gamma=best[2],class_weight=best[3])
        model.fit(np.vstack([Xext,Xlocal[tr]]),np.concatenate([yext,ylocal[tr]]))
        pred[va]=model.predict(Xlocal[va])
        fold_records.append({"fold":fold,"C":best[1],"gamma":best[2],"class_weight":best[3],
             "inner_macro_f1":best[0],"outer_macro_f1":f1_score(ylocal[va],pred[va],average="macro",zero_division=0)})
        print("FOLD",fold,"params",best,"macroF1",round(fold_records[-1]["outer_macro_f1"],4),flush=True)
    assert (pred>=0).all()
    scores={"external_only_macro_f1":f1_score(ylocal,direct,average="macro",zero_division=0),
            "external_only_accuracy":accuracy_score(ylocal,direct),
            "external_plus_local_cv_macro_f1":f1_score(ylocal,pred,average="macro",zero_division=0),
            "external_plus_local_cv_accuracy":accuracy_score(ylocal,pred),
            "external_plus_local_recalls":recall_score(ylocal,pred,labels=np.arange(5),average=None,zero_division=0).tolist(),
            "confusion_matrix":confusion_matrix(ylocal,pred,labels=np.arange(5)).tolist(),
            "folds":fold_records,"external_audit":metadata,
            "warning":"This is extra supervised SST5 training data. Course may prohibit external data. Do not describe as 511-label-only result."}
    (out/"report.json").write_text(json.dumps(scores,indent=2))
    pd.DataFrame({"PhraseId":train.PhraseId,"Sentiment":ylocal,
                  "prediction_augmented":pred,"prediction_external_only":direct}).to_csv(out/"oof_predictions.csv",index=False)
    print("TOTAL extra-data nested OOF macroF1",round(scores["external_plus_local_cv_macro_f1"],4),
          "accuracy",round(scores["external_plus_local_cv_accuracy"],4),flush=True)


def main():
    global train
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",default=str(ROOT/"train.csv"))
    p.add_argument("--test-text",default=str(ROOT/"test.csv"))
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--near-duplicate-threshold",type=int,default=90)
    p.add_argument("--output",default=str(ROOT/"experiments/results/external_sst5"))
    args=p.parse_args()
    if not 70<=args.near_duplicate_threshold<=100:
        raise ValueError("Near-duplicate threshold out of range")
    train=pd.read_csv(args.train,usecols=["PhraseId","Phrase","Sentiment"])
    test=pd.read_csv(args.test_text,usecols=["Phrase"])
    if not train.PhraseId.is_unique or train.Phrase.isna().any() or test.Phrase.isna().any():
        raise ValueError("Bad local rows")
    Xlocal=load_embedding(F/"bge_small_train.npy",train)
    ylocal=train.Sentiment.to_numpy(dtype=np.int64)
    texts,labels,audit=prepare_external(train,test,args.near_duplicate_threshold)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    (out/"audit.json").write_text(json.dumps(audit,indent=2))
    print("EXTERNAL AUDIT:",json.dumps(audit),flush=True)
    Xext=encode(texts,args)
    np.save(out/"external_embeddings.npy",Xext)
    np.save(out/"external_labels.npy",labels)
    cv(args,Xext,labels,Xlocal,ylocal,out,audit)


if __name__=="__main__":
    main()
