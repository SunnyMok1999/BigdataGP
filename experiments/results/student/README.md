# Projection-student audit

This is a new implementation inspired by the supplied ChatGPT description; its
unavailable code, targets, splits, and hyperparameters cannot be reproduced
exactly. Only train.csv and frozen MiniLM features in that same row order are
read. The original test partition is never opened.

Each outer holdout is excluded from scaling, early stopping, fitting, and
teacher targets used for training. A stratified 20% inner validation selects an
epoch by cross entropy; the student is then fitted from scratch on the entire
outer training fold for exactly that many epochs. The four variants and all
training settings are specified in student_distillation.py and summary.json.

Teacher targets explicitly inject true sentiment labels; their silhouette is a
construction diagnostic, not evidence of generalization. Out-of-fold embeddings
come from separate fitted students. Shared fixed prototype coordinates prevent
arbitrary rotations, but pooled OOF silhouette is still a diagnostic; consult
per-fold silhouettes too. PCA panels have separate axes and do not establish
class separability beyond the quantitative metrics.

The full_training_model.pt files use all 511 training rows and median selected
epochs. Their in-sample training accuracy is not an unbiased performance estimate.
The separate memorization_control directory deliberately trains teacher+CE on
all rows for 150 epochs. Its scores and geometry use those same training rows;
they demonstrate memorization capacity and cannot establish generalization.
The OOF metrics estimate each prespecified training procedure; choosing the best
of these variants after seeing the scores can be optimistic. No significance or
independent test-performance claim is made. Baselines and encoder experiments
must be compared on identical rows, labels, and held-out folds.

For inference on new MiniLM embeddings (same encoder and text preprocessing):

    python experiments/student_distillation.py --predict-features new_minilm.npy \
        --checkpoint experiments/results/student/teacher_ce/full_training_model.pt \
        --prediction-output predictions.npz

The prediction archive contains embeddings, probabilities and predicted labels;
no true labels are needed for inference.
