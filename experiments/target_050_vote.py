"""Pre-registered exploratory vote across prior independent OOF predictors.

Uses labels *only for evaluation*. Voting rule fixed before loading outputs.
Reported score is exploratory: component model families previously selected
using the same training CV, so family-selection optimism remains.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def single_file(root, suffix):
    matches = list(Path(root).rglob(suffix))
    if len(matches) != 1:
        raise ValueError(f"Expected one {suffix} beneath {root}, got {matches}")
    return matches[0]


def selected_predictions(path, candidate, label_column="prediction"):
    df = pd.read_csv(path)
    if candidate is not None:
        if "candidate" not in df:
            raise ValueError("Missing candidate column")
        df = df.loc[df.candidate.eq(candidate)]
    if not df.PhraseId.is_unique:
        raise ValueError("Duplicate IDs")
    return df.set_index("PhraseId")[label_column]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--amazon",required=True)
    parser.add_argument("--external",required=True)
    parser.add_argument("--cached",required=True)
    parser.add_argument("--out",required=True)
    a=parser.parse_args()
    base=Path(__file__).resolve().parents[1]
    train=pd.read_csv(base/"train.csv")
    if not train.PhraseId.is_unique:
        raise ValueError("Duplicate training IDs")
    ids=train.PhraseId.tolist()
    y=train.Sentiment.to_numpy(dtype=int)
    a_pred=selected_predictions(
        single_file(a.amazon, "amazon_roberta5_hidden_cv/oof_predictions.csv"),
        "sent_hidden_svm"
    ).reindex(ids)
    e_pred=selected_predictions(
        single_file(a.external, "oof_predictions.csv"),None,"prediction_augmented"
    ).reindex(ids)
    b_pred=selected_predictions(
        single_file(a.cached, "oof_predictions.csv"),"bge_svm"
    ).reindex(ids)
    matrix=np.column_stack([a_pred,e_pred,b_pred]).astype(float)
    if not np.isfinite(matrix).all() or not np.isin(matrix,np.arange(5)).all():
        raise ValueError("Incompatible predictions or row alignment")
    matrix=matrix.astype(int)
    # If all three disagree, prefer the strongest preexisting Amazon
    # source; no evaluation-label access in tie-break or model selection.
    votes=np.array([
        (np.bincount(row,minlength=5).argmax() if np.bincount(row,minlength=5).max()>1
         else row[0])
        for row in matrix
    ])
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows={"PhraseId":ids,"Sentiment":y,
          "amazon_hidden_svm":matrix[:,0],
          "external_sst5":matrix[:,1],"bge_svm":matrix[:,2],
          "fixed_3vote_prediction":votes}
    pd.DataFrame(rows).to_csv(out/"oof_votes.csv",index=False)
    scores={}
    for name,pred in [("amazon_hidden_svm",matrix[:,0]),
                      ("external_sst5",matrix[:,1]),("bge_svm",matrix[:,2]),
                      ("fixed_three_model_vote",votes)]:
        scores[name]={"macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
                     "accuracy":float(accuracy_score(y,pred)),
                     "confusion_matrix":confusion_matrix(y,pred,labels=np.arange(5)).tolist(),
                     "classification_report":classification_report(y,pred,labels=np.arange(5),output_dict=True,zero_division=0)}
        print(f"{name}: macroF1={scores[name]['macro_f1']:.4f} accuracy={scores[name]['accuracy']:.4f}",flush=True)
    scores["note"]="Exploratory predeclared hard-label vote; source model choice already informed by training OOF and external SST-5 labeled supervision was used. Not independent validation."
    (out/"report.json").write_text(json.dumps(scores,indent=2))


if __name__=="__main__":
    main()
