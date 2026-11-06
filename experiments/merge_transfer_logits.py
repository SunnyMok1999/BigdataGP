"""Join two train-only pretrained sentiment predictions with exact row/text checks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def join(paths,output):
    metadata=[json.loads(p.with_suffix(".metadata.json").read_text()) for p in paths]
    sig={m["phrase_id_and_text_sha256"] for m in metadata}
    if len(sig)!=1: raise ValueError("Models were run on different review texts/order")
    arrays=[np.load(p,allow_pickle=False) for p in paths]
    rows=[pd.read_csv(p.with_suffix(".rows.csv")) for p in paths]
    for r in rows[1:]:
        if not rows[0].equals(r): raise ValueError("PhraseId alignment mismatch")
    if sum(x.shape[1] for x in arrays)!=8: raise ValueError("Expected one 5-class and one 3-class model")
    x=np.concatenate(arrays,axis=1).astype(np.float32)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.save(output,x)
    rows[0].to_csv(output.with_suffix(".rows.csv"),index=False)
    output.with_suffix(".metadata.json").write_text(json.dumps({
        "phrase_id_and_text_sha256": metadata[0]["phrase_id_and_text_sha256"],
        "alias":"review5_plus_tweet3",
        "inputs":[{"model":m["model"],"shape":m["shape"]} for m in metadata],
        "shape":list(x.shape),
        "note":"No local labels used in extraction/merge. Off-domain product and Twitter training; no SST5 explicit fine-tuning."
    },indent=2))
    print(f"merged {output}: {x.shape}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--inputs",nargs=2,type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    join(a.inputs,a.output)
