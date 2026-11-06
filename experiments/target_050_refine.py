"""Fine-grained, training-only nested CV near a 0.50 macro-F1 target."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[1]


def build(name):
    if name=="scale_l2":
        est=Pipeline([("scale",StandardScaler()),("l2",Normalizer()),
                      ("svc",SVC(kernel="rbf"))])
        grid={"svc__C":[.3,1,3,10,30],
              "svc__gamma":["scale",.1,.5,1,2],
              "svc__class_weight":[None,"balanced"]}
    elif name=="scaled_fine":
        est=Pipeline([("scale",StandardScaler()),("svc",SVC(kernel="rbf"))])
        grid={"svc__C":[.3,1,3,10,30],
              "svc__gamma":["scale",.0003,.0006,.001,.002],
              "svc__class_weight":[None,"balanced"]}
    elif name=="scaled_balanced_fine":
        est=Pipeline([("scale",StandardScaler()),("svc",SVC(kernel="rbf"))])
        # Fixed, label-independent relative adjustments to balance class 0/2/4
        # around sklearn's train-fold-only balanced class weights.
        grid={"svc__C":[.3,1,3,10],
              "svc__gamma":["scale",.0003,.001,.003],
              "svc__class_weight":["balanced",None,
                  {0:2.5,1:1.,2:1.4,3:1.,4:2.},
                  {0:2.,1:1.,2:1.6,3:1.,4:1.7}]}
    else:raise ValueError(name)
    return est,grid


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="experiments/features/amazon_roberta5_hidden_train.npy")
    p.add_argument("--models",nargs="+",default=["scale_l2","scaled_fine","scaled_balanced_fine"])
    p.add_argument("--out",default="experiments/results/target_050_refinement")
    a=p.parse_args()
    df=pd.read_csv(ROOT/"train.csv")
    features=Path(a.input)
    x=np.load(features,allow_pickle=False).astype(float)
    if x.shape[0]!=len(df) or not np.isfinite(x).all():raise ValueError("Rows/finite mismatch")
    rows=pd.read_csv(features.with_name(features.stem+"_rows.csv"))
    if not np.array_equal(rows.PhraseId,df.PhraseId):raise ValueError("PhraseId alignment")
    y=df.Sentiment.to_numpy(dtype=int)
    if set(y)!=set(range(5)):raise ValueError("Expected five classes")
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    leaderboard=[]
    for name in a.models:
        est,grid=build(name)
        oof=np.full(len(y),-1,dtype=int);folds=[]
        for fold,(tr,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(x,y)):
            gs=GridSearchCV(est,grid,cv=StratifiedKFold(3,shuffle=True,random_state=42+fold),
                            scoring="f1_macro",n_jobs=1,error_score="raise",refit=True)
            gs.fit(x[tr],y[tr])
            oof[va]=gs.predict(x[va])
            folds.append({"fold":fold,"best_params":gs.best_params_,"inner_f1":float(gs.best_score_),
                          "outer_f1":float(f1_score(y[va],oof[va],average="macro",zero_division=0))})
            print(name,"fold",fold,"macroF1",round(folds[-1]["outer_f1"],4),"best",gs.best_params_,flush=True)
        score={"name":name,"macro_f1":float(f1_score(y,oof,average="macro",zero_division=0)),
               "accuracy":float(accuracy_score(y,oof)),
               "recalls":recall_score(y,oof,labels=np.arange(5),average=None,zero_division=0).tolist(),
               "confusion":confusion_matrix(y,oof,labels=np.arange(5)).tolist(),
               "folds":folds}
        leaderboard.append(score)
        pd.DataFrame({"PhraseId":df.PhraseId,"true":y,"prediction":oof}).to_csv(out/f"{name}_oof.csv",index=False)
        print("RESULT",name,"macroF1",round(score["macro_f1"],4),"accuracy",round(score["accuracy"],4),flush=True)
    (out/"details.json").write_text(json.dumps(leaderboard,indent=2))
    pd.DataFrame([{k:v for k,v in s.items() if k not in ("recalls","confusion","folds")}
                  for s in leaderboard]).sort_values("macro_f1",ascending=False).to_csv(out/"leaderboard.csv",index=False)


if __name__=="__main__":main()
