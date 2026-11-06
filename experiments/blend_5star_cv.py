"""Fold-safe blend of BGE RBF SVM + off-domain product-review 5-star logits.

Tuning of RBF params, blend and optional per-class decision biases occurs ONLY
within each outer training fold. Outer validation labels never tune thresholds.
All results are exploratory training-only cross-validation. No test.csv loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

from strong_sentiment_cv import ROOT, F, load_embedding, load_transfer


def tuned_probs(model_probs, external_logits, alpha, temperature, class_bias=None):
    ext = softmax(external_logits / temperature, axis=1)
    pred = (1-alpha) * model_probs + alpha * ext
    if class_bias is not None:
        pred = np.log(np.clip(pred, 1e-7, 1)) + class_bias
    return pred


def choose_bias(probs, y, seed):
    """Greedy low-capacity class-prior shifts, fitted only on train OOF.

    Keep class 2 as reference (identifiability); one global scale selected
    by nested fold score. Small grid to reduce overfit.
    """
    best = np.zeros(5)
    bestscore = f1_score(y, probs.argmax(1), average="macro", zero_division=0)
    logp = np.log(np.clip(probs,1e-7,1))
    # Penalize substantial shifts; only search at most 2 brief passes.
    for _ in range(2):
        improved=False
        for cls in (0,1,3,4):
            for step in (-.4,-.2,0,.2,.4):
                b = best.copy()
                b[cls] = step
                f1 = f1_score(y,(logp+b).argmax(1),average="macro",zero_division=0)
                if f1 > bestscore + .007:
                    bestscore=f1
                    best=b
                    improved=True
        if not improved: break
    return best


def evaluate(args):
    data=pd.read_csv(args.train)
    y=data.Sentiment.to_numpy(dtype=int)
    X=load_embedding(F/"bge_small_train.npy", data)
    ext, metadata=load_transfer(Path(args.transfer), data)
    if ext.shape[1]!=5:
        raise ValueError("This zero-shot blending approach requires five-class transfer logits")
    folds=list(StratifiedKFold(5, shuffle=True, random_state=42).split(X,y))
    records=[]
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    probs_oof=np.zeros((len(y),5))
    pred=np.zeros(len(y),dtype=int)
    for i,(tr,va) in enumerate(folds):
        base=SVC(kernel="rbf",probability=True,random_state=42+i)
        grid={"C":[.1,1.,10.,100.],"gamma":[.1,.5,1.,2.],
              "class_weight":[None,"balanced"]}
        search=GridSearchCV(base,grid,scoring="f1_macro",cv=StratifiedKFold(3,shuffle=True,random_state=42+i),
                            n_jobs=1,refit=True)
        search.fit(X[tr],y[tr])
        chosen=search.best_params_
        model=SVC(kernel="rbf",probability=True,random_state=42+i,**chosen)
        # OOF predictions from inner partitions. No outer validation labels.
        innerprob=cross_val_predict(model,X[tr],y[tr],
             cv=StratifiedKFold(3,shuffle=True,random_state=100+i),method="predict_proba")
        alphaT=[]
        for alpha in (0.,.2,.4,.6,.8,1.):
            for temp in (.7,1.,1.5,2.):
                p=tuned_probs(innerprob,ext[tr],alpha,temp)
                score=f1_score(y[tr],p.argmax(1),average="macro",zero_division=0)
                alphaT.append((score,alpha,temp))
        alphaT.sort(key=lambda x:(-x[0],x[1],abs(x[2]-1)))
        _,alpha,temp=alphaT[0]
        inner_blended=tuned_probs(innerprob,ext[tr],alpha,temp)
        bias=choose_bias(inner_blended,y[tr],i) if args.bias else np.zeros(5)
        model.fit(X[tr],y[tr])
        val_blended=tuned_probs(model.predict_proba(X[va]),ext[va],alpha,temp,bias)
        thispred=val_blended.argmax(1)
        pred[va]=thispred
        probs_oof[va]=softmax(val_blended,axis=1)
        rows={"fold":i,"alpha":alpha,"external_temperature":temp,
              "bias":bias.tolist(),"inner_blend_f1":alphaT[0][0],
              "best_base_params":chosen,"outer_accuracy":accuracy_score(y[va],thispred),
              "outer_macro_f1":f1_score(y[va],thispred,average="macro",zero_division=0)}
        records.append(rows)
        print("fold",i,"params",chosen,"alpha",alpha,"temp",temp,"bias",bias.tolist(),
              "outer_macroF1",round(rows["outer_macro_f1"],4),flush=True)
    overall={"macro_f1":f1_score(y,pred,average="macro",zero_division=0),
             "accuracy":accuracy_score(y,pred),
             "recalls":recall_score(y,pred,labels=np.arange(5),average=None,zero_division=0).tolist(),
             "confusion_matrix":confusion_matrix(y,pred,labels=np.arange(5)).tolist(),
             "folds":records,"transfer_metadata":metadata,
             "note":"five outer / three inner; CV tuned blend and bias, test.csv untouched; selecting this method among many still optimism"}
    pd.DataFrame([{"PhraseId":int(id_),"fold":int(next(i for i,(_,v) in enumerate(folds) if row in v)),
                  "prediction":int(pred[row]),**{f"prob_{j}":float(probs_oof[row,j]) for j in range(5)}}
                 for row,id_ in enumerate(data.PhraseId)]).to_csv(out/"oof_predictions.csv",index=False)
    (out/"metrics.json").write_text(json.dumps(overall,indent=2))
    print(f"TOTAL nested OOF macroF1={overall['macro_f1']:.4f} ACC={overall['accuracy']:.4f}",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train",default=str(ROOT/"train.csv"))
    p.add_argument("--transfer",required=True)
    p.add_argument("--bias",action="store_true")
    p.add_argument("--output",required=True)
    evaluate(p.parse_args())
