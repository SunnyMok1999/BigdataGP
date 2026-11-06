#!/usr/bin/env python3
"""Report Experiment A acquisition + Experiment B encoder ablations, TRAIN only.

Output is descriptive. Selecting a new model using OOF performance still
incurs selection optimism. The separate test.csv is NEVER opened here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def acquisition_report(curves: pd.DataFrame, output: Path):
    required = {"seed", "policy", "round", "labeled", "macro_f1", "accuracy"}
    if not required.issubset(curves):
        raise ValueError("Incomplete active-learning curves")
    records, areas = [], []
    for (seed, policy), block in curves.groupby(["seed", "policy"]):
        block = block.sort_values("labeled")
        final = block.iloc[-1]
        records.append(dict(seed=seed, policy=policy, budget=int(final.labeled),
                            macro_f1=float(final.macro_f1), accuracy=float(final.accuracy),
                            **{key:float(final[key]) for key in
                               [f"recall_{i}" for i in range(5)] if key in block}))
        if len(block) > 1:
            x, y = block.labeled.to_numpy(float), block.macro_f1.to_numpy(float)
            areas.append(dict(seed=seed, policy=policy,
                              macro_f1_auc=float(np.trapz(y, x)/(x[-1]-x[0]))))
    final = pd.DataFrame(records)
    final.to_csv(output / "final_by_seed.csv", index=False)
    agg = final.groupby("policy").agg(
        accuracy=("accuracy", "mean"), macro_f1=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        **{f"recall_{i}":(f"recall_{i}","mean") for i in range(5)
           if f"recall_{i}" in final})
    agg.to_csv(output / "acquisition_aggregate.csv")
    pd.DataFrame(areas).groupby("policy").macro_f1_auc.mean().to_csv(output / "auc_aggregate.csv")
    paired=[]
    if "random" in final.policy.unique():
        random=final[final.policy.eq("random")].set_index("seed").macro_f1
        for policy, sub in final.groupby("policy"):
            aligned=sub.set_index("seed").macro_f1
            if not aligned.index.equals(random.index):
                aligned=aligned.reindex(random.index)
            gain=(aligned-random).dropna()
            se=float(gain.std(ddof=1)/np.sqrt(len(gain))) if len(gain)>1 else float("nan")
            paired.append(dict(policy=policy,n_seeds=len(gain),paired_f1_gain=float(gain.mean()),
                               wins=int((gain>0).sum()),
                               ci95_low=float(gain.mean()-1.96*se),
                               ci95_high=float(gain.mean()+1.96*se)))
        pd.DataFrame(paired).to_csv(output / "paired_vs_random.csv",index=False)
    print("Experiment A final budget — mean across seeds")
    print(agg[["accuracy","macro_f1","macro_f1_sd"]].sort_values("macro_f1",ascending=False).round(4))
    return agg


def representation_report(folder:Path, output:Path):
    variants = ["minilm_ce","minilm_supcon","minilm_ordinal",
                "minilm_adjacent_v2","minilm_adjacent_light",
                "minilm_adjacent_strong","minilm_adjacent_no_triplet",
                "bge_ce","bge_adjacent_v2"]
    records=[]
    for name in variants:
        path=folder/name/"summary.json"
        if not path.exists():
            records.append(dict(model=name,status="not_run"))
            continue
        data=json.loads(path.read_text())
        conf=np.asarray(data.get("confusion_matrix",[]))
        row=dict(model=name,status="completed",
                 accuracy=data.get("accuracy"),macro_f1=data.get("macro_f1"),
                 class0_recall=data.get("class0_recall"),mae=data.get("mae"))
        if conf.shape==(5,5):
            for i in range(5):
                row[f"recall_{i}"]=float(conf[i,i]/max(1,conf[i].sum()))
        records.append(row)
    results=pd.DataFrame(records)
    results.to_csv(output/"representation_ablation.csv",index=False)
    print("\nExperiment B — saved 5-fold encoder runs")
    print(results[["model","status","macro_f1","accuracy","class0_recall"]].to_string(index=False))
    print("\nScores here are training-only OOF; no test-set tuning.")
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--curves",type=Path,default=ROOT/"experiments/results/active_learning_new/curves.csv")
    p.add_argument("--encoder",type=Path,default=ROOT/"experiments/results/encoder")
    p.add_argument("--out",type=Path,default=ROOT/"experiments/results/experiment_AB_report")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.curves.exists():
        acquisition_report(pd.read_csv(args.curves),args.out)
    else:
        print("Experiment A pending:",args.curves)
    representation_report(args.encoder,args.out)


if __name__=="__main__":
    main()
