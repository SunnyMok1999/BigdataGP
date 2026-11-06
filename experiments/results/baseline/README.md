# Baseline tuning

Run from the project root:

```sh
/opt/anaconda3/envs/dml/bin/python experiments/baseline_tuning.py
```

This experiment reads only `Cleaned data/cleaned_train.csv`. It never opens the supplied test partition. Five stratified folds (shuffle seed 42) are shared by all candidates. TF-IDF is fitted inside each training fold. Every training row has a unique SentenceId and unique text, checked before CV.

The 48 candidates compare StemmedText and CleanText using C in {0.1, 1, 10}, unigram/bigram features, min_df in {1, 2}, class_weight in {None, balanced}, and sublinear_tf=True. The original default StemmedText baseline is evaluated separately.

`cv_candidates.csv` includes all candidate and fold scores. `cv_folds.csv` records row assignments. `selected_configuration.json` records selection, metrics, and the confusion matrix. `selected_model_oof_predictions.csv` contains selected-model out-of-fold predictions. Because the same CV folds selected the model, these scores have selection optimism; they are not an independent final evaluation.

`selected_model.joblib` contains a dictionary with `pipeline` and `representation`; the pipeline is fitted on all 511 training rows. For later evaluation, load the dictionary and call `bundle['pipeline'].predict(frame[bundle['representation']])`. Use only trusted joblib files.

Dependencies: numpy, pandas, scipy, scikit-learn, joblib, threadpoolctl. Exact installed versions and runtime are in `run_metadata.json`. Parallelism is limited to two workers and one numerical thread each.
