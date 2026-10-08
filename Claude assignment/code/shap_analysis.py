"""
MLA final project: SHAP values for the random forest (Part 1b).

1. Explains the final random forest (200 trees, the same model as in
   run_analysis.py) with shap.TreeExplainer on 1,000 test images (100 per class).
   This gives Figure 4 of the report. Explaining test images is fine here: nothing
   is chosen based on them.
2. Pixel selection: on the first CV fold, a 50-tree forest is explained on 1,000
   held-out images of that fold; the pixels are ranked by mean |SHAP| and forests
   on only the top-k pixels are scored on the other held-out images of the fold
   (so the images used for ranking are not used for scoring).

The SHAP computation is split over all CPU cores.
Run from the code/ folder after run_analysis.py:   python shap_analysis.py
Writes: ../results/shap.json and ../figures/importance.png
"""
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tensorflow import keras

RNG = 42
N_PER_CLASS = 100
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
RES_DIR = os.path.join(HERE, "..", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


# Load Fashion-MNIST from Keras (downloads on first run if not cached)
(x_trainset, y_trainset), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()


def square2row(square, num_obs):
    """Reshape a stack of 28x28 images into (num_obs, 784) flat vectors."""
    return np.resize(square, [num_obs, 28 * 28])


X_train = square2row(x_trainset, x_trainset.shape[0]).astype(np.float32) / 255.0
y_train = y_trainset
X_test = square2row(x_test, x_test.shape[0]).astype(np.float32) / 255.0


def _shap_chunk(model, X):
    sv = np.asarray(shap.TreeExplainer(model).shap_values(X, check_additivity=False))
    if sv.shape[-1] == 10:              # (n_images, 784, 10) -> (10, n_images, 784)
        sv = np.moveaxis(sv, -1, 0)
    return sv


def shap_values(model, X, chunk=25):
    """SHAP values of a forest for the images in X, computed in parallel chunks.
    Returns an array of shape (10 classes, n_images, 784)."""
    parts = Parallel(n_jobs=-1)(delayed(_shap_chunk)(model, X[i:i + chunk])
                                for i in range(0, len(X), chunk))
    return np.concatenate(parts, axis=1)


def pick(y, n):
    """Indices of the first n images of every class."""
    return np.concatenate([np.where(y == c)[0][:n] for c in range(10)])


out = {"n_images_per_class": N_PER_CLASS}

# ---------------------------------------------------------------------
# 1. Final forest, explained on 1,000 test images
# ---------------------------------------------------------------------
log("fitting the final random forest (200 trees)")
rf = RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=RNG)
rf.fit(X_train, y_train)
idx = pick(y_test, N_PER_CLASS)
log(f"SHAP values for {len(idx)} test images")
sv = shap_values(rf, X_test[idx])
mean_abs = np.abs(sv).mean(axis=(0, 1))
shirt = sv[6][y_test[idx] == 6].mean(axis=0)          # push towards "Shirt" on true shirts
mdi = rf.feature_importances_
order = np.argsort(mean_abs)[::-1]
out["final_forest"] = {
    "n_images": int(len(idx)),
    "corr_shap_vs_impurity": round(float(np.corrcoef(mean_abs, mdi)[0, 1]), 3),
    "share_of_mean_abs_shap_in_top_100": round(float(mean_abs[order[:100]].sum() / mean_abs.sum()), 3),
    "top20_pixels_row_col": [[int(p // 28), int(p % 28)] for p in order[:20]],
}
log(f"final forest: {out['final_forest']}")

# ---------------------------------------------------------------------
# 2. Pixel selection on the first CV fold (same folds as run_analysis.py)
# ---------------------------------------------------------------------
i_tr, i_va = next(StratifiedKFold(n_splits=3, shuffle=True, random_state=RNG).split(X_train, y_train))
X_a, y_a, X_b, y_b = X_train[i_tr], y_train[i_tr], X_train[i_va], y_train[i_va]
rank_idx = pick(y_b, N_PER_CLASS)                      # held-out images used for ranking
score_mask = np.ones(len(y_b), dtype=bool)
score_mask[rank_idx] = False                           # ... and the rest for scoring
log("fold 1: 50-tree forest and SHAP ranking")
rf_fold = RandomForestClassifier(n_estimators=50, max_features="sqrt", n_jobs=-1, random_state=RNG)
rf_fold.fit(X_a, y_a)
ranking = np.argsort(np.abs(shap_values(rf_fold, X_b[rank_idx])).mean(axis=(0, 1)))[::-1]
fs = {"784": round(accuracy_score(y_b[score_mask], rf_fold.predict(X_b[score_mask])), 4)}
for k in [50, 100, 200, 400]:
    cols = ranking[:k]
    model = RandomForestClassifier(n_estimators=50, max_features="sqrt", n_jobs=-1,
                                   random_state=RNG).fit(X_a[:, cols], y_a)
    fs[str(k)] = round(accuracy_score(y_b[score_mask], model.predict(X_b[score_mask][:, cols])), 4)
    log(f"  top {k} pixels: {fs[str(k)]}")
out["pixel_selection"] = {"n_scored_images": int(score_mask.sum()), "accuracy": fs}

# ---------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "savefig.bbox": "tight"})
class_means = np.stack([X_train[y_train == c].mean(axis=0) for c in range(10)])
fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
panels = [
    (mdi.reshape(28, 28), "inferno", False, "(a) Impurity decrease"),
    (mean_abs.reshape(28, 28), "inferno", False, "(b) Mean |SHAP|, all classes"),
    (shirt.reshape(28, 28), "RdBu_r", True, "(c) SHAP for 'Shirt' on shirts"),
    (np.abs(class_means[6] - class_means[2]).reshape(28, 28), "inferno", False,
     "(d) |mean Shirt $-$ mean Pullover|"),
]
for ax, (img, cmap, sym, title) in zip(axes, panels):
    if sym:
        v = np.abs(img).max()
        im = ax.imshow(img, cmap=cmap, vmin=-v, vmax=v)
    else:
        im = ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    cb = plt.colorbar(im, ax=ax, fraction=0.046)
    cb.ax.tick_params(labelsize=7)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "importance.png"), dpi=200)
plt.close()

with open(os.path.join(RES_DIR, "shap.json"), "w") as f:
    json.dump(out, f, indent=2)
log("done")
