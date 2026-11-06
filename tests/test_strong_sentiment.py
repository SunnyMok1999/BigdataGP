"""Smoke/invariant tests for stronger sentiment CV; no test.csv loaded."""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from strong_sentiment_cv import BlockScaler, load_embedding, load_transfer, candidate_spec

def test_cached_train_embeddings_align():
    df=pd.read_csv(ROOT/"train.csv")
    a=load_embedding(ROOT/"experiments/features/bge_small_train.npy",df)
    assert a.shape==(511,384) and np.isfinite(a).all()
    corrupted=df.copy()
    corrupted.loc[0,"PhraseId"]+=1
    with pytest.raises(ValueError,match="Row mismatch"):
        load_embedding(ROOT/"experiments/features/bge_small_train.npy",corrupted)

def test_block_scaler_fit_uses_train_only():
    rng=np.random.default_rng(7)
    a=rng.normal(size=(20,384))
    b=rng.normal(size=(20,5))
    X=np.hstack([a,b])
    fit=BlockScaler(width=384,extra_weight=2.).fit(X[:12])
    x1=fit.transform(X[12:])
    altered=X.copy()
    altered[12:,384:]+=10000
    fit2=BlockScaler(width=384,extra_weight=2.).fit(altered[:12])
    assert np.array_equal(fit.extra_scaler_.mean_,fit2.extra_scaler_.mean_)
    assert np.allclose(fit.transform(X[12:]),fit2.transform(X[12:]))
    assert np.allclose(np.linalg.norm(x1[:,384:],axis=1),2.)

def test_candidates_do_not_look_at_labels_to_build_features():
    rng=np.random.default_rng(3)
    x=rng.normal(size=(18,384))
    t=rng.normal(size=(18,5))
    for name in ("bge_lr","bge_svm","bge_linear","transfer_direct",
                 "transfer_only_lr","bge_transfer_lr","bge_transfer_svm"):
        transformed,est,grid=candidate_spec(name,x,None,t)
        assert transformed.shape[0]==18
        if name=="transfer_direct":
            assert est is None
        else:
            assert est is not None and grid is not None

def test_transfer_metadata_enforces_training_text(tmp_path):
    df=pd.DataFrame({"PhraseId":[1,2],"Phrase":["awful","great"],"Sentiment":[0,4]})
    arr=np.ones((2,5),dtype=np.float32)
    path=tmp_path/"review5_train.npy"
    np.save(path,arr)
    df[["PhraseId"]].to_csv(path.with_suffix(".rows.csv"),index=False)
    import hashlib
    h=hashlib.sha256()
    for k,v in zip(df.PhraseId,df.Phrase):
        h.update((str(int(k))+"\t"+v+"\n").encode())
    path.with_suffix(".metadata.json").write_text(json.dumps({"phrase_id_and_text_sha256":h.hexdigest()}))
    x,_=load_transfer(path,df)
    assert x.shape==(2,5)
    df.loc[1,"Phrase"]="different"
    with pytest.raises(ValueError,match="Text mismatch"):
        load_transfer(path,df)
