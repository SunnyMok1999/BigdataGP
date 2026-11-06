"""Nested five-class sentiment optimization of off-domain Amazon RoBERTa CLS vectors.

The pretrained model was trained on Amazon review sentiment, NOT on the local
five-class label fold. Exact row-ordered cached vectors are checked by PhraseId.
No local test.csv access. Family selection across OOF arms is exploratory.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]

def candidates(name):
    if name=="rbf_dense":
        return SVC(kernel="rbf"),{"C":[.1,1,3,10,30,100],
            "gamma":[.05,.1,.25,.5,1,2,4,8],"class_weight":[None,"balanced"]}
    if name=="poly2":
        return SVC(kernel="poly",degree=2,coef0=1),{"C":[.1,1,10,100],
            "gamma":[.1,.5,1,2],"class_weight":[None,"balanced"]}
    if name=="poly3":
        return SVC(kernel="poly",degree=3,coef0=1),{"C":[.1,1,10,100],
            "gamma":[.1,.5,1,2],"class_weight":[None,"balanced"]}
    if name=="linear":
        return SVC(kernel="linear"),{"C":[.01,.1,1,10,100],
             "class_weight":[None,"balanced"]}
    if name=="scaled_rbf":
        return Pipeline([("scaler",StandardScaler()),("svc",SVC(kernel="rbf"))]),{
            "svc__C":[.1,1,10,100],"svc__gamma":["scale",.0001,.001,.01],
            "svc__class_weight":[None,"balanced"]}
    if name=="logistic":
        return LogisticRegression(max_iter=2500),{"C":[.01,.1,1,10,100],
             "class_weight":[None,"balanced"]}
    raise ValueError(name)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="experiments/features/amazon_roberta5_hidden_train.npy")
    p.add_argument("--models",nargs="+",default=["rbf_dense","poly2","poly3","linear","scaled_rbf","logistic"])
    p.add_argument("--out",default="experiments/results/target_050_amazon")
    a=p.parse_args()
    source=Path(a.input)
    train=pd.read_csv(ROOT/"train.csv")
    rows=pd.read_csv(source.with_name(source.stem+"_rows.csv"))
    if not train.PhraseId.is_unique or not np.array_equal(train.PhraseId,rows.PhraseId):
        raise ValueError("PhraseId misalignment")
    if "Sentiment" in rows and not np.array_equal(train.Sentiment, rows.Sentiment):
        raise ValueError("Sentiment misalignment")
    X=np.load(source,allow_pickle=False).astype(np.float64)
    if X.shape[0]!=len(train) or not np.isfinite(X).all():
        raise ValueError("Invalid pretrained feature matrix")
    y=train.Sentiment.to_numpy(int)
    if set(y)!=set(range(5)):raise ValueError("Five classes required")
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    report=[]
    for name in a.models:
        est,grid=candidates(name)
        pred=np.full(len(y),-1,dtype=int)
        parameters=[]
        for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
            cv=StratifiedKFold(3,shuffle=True,random_state=42+fold)
            search=GridSearchCV(est,grid,cv=cv,scoring="f1_macro",n_jobs=1,error_score="raise",refit=True)
            search.fit(X[tr],y[tr])
            pred[va]=search.predict(X[va])
            f1=f1_score(y[va],pred[va],average="macro",zero_division=0)
            print(name,"fold",fold,"macroF1",round(f1,4),"params",search.best_params_,flush=True)
            parameters.append({"fold":fold,"params":search.best_params_,"inner_f1":float(search.best_score_),"outer_f1":float(f1)})
        score={"model":name,"macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
               "accuracy":float(accuracy_score(y,pred)),
               "recall":recall_score(y,pred,labels=np.arange(5),average=None,zero_division=0).tolist(),
               "confusion":confusion_matrix(y,pred,labels=np.arange(5)).tolist(),
               "folds":parameters}
        report.append(score)
        pd.DataFrame({"PhraseId":train.PhraseId,"label":y,"prediction":pred}).to_csv(out/(name+"_oof.csv"),index=False)
        print("RESULT",name,"macroF1",round(score["macro_f1"],4),"accuracy",round(score["accuracy"],4),flush=True)
    (out/"details.json").write_text(json.dumps(report,indent=2))
    pd.DataFrame([{k:v for k,v in z.items() if k not in ("recall","confusion","folds")}
                   for z in report]).sort_values("macro_f1",ascending=False).to_csv(out/"leaderboard.csv",index=False)


if __name__=="__main__":main()
