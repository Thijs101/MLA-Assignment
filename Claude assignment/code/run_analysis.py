"""
MLA final project: Fashion-MNIST with decision trees and tree ensembles (Part 1).

Uses numpy, matplotlib, scikit-learn
(DecisionTreeClassifier, BaggingClassifier, RandomForestClassifier, KMeans,
GridSearchCV). SHAP values are in shap_analysis.py. The gradient boosting reference model
is in gradient_boosting.py because it takes much longer to train.

Produces:
  ../results/metrics.json     every Part 1 number quoted in the report
  ../figures/*.png            every figure used in the report

Run from the code/ folder (Fashion-MNIST is downloaded through Keras on the first run):
    python run_analysis.py
"""
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)

from tensorflow import keras

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

RNG = 42
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
RES_DIR = os.path.join(HERE, "..", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

UPPER = [0, 2, 4, 6]  # T-shirt/top, Pullover, Coat, Shirt
SHORT = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
         "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.bbox": "tight",
})
C1, C2, C3, C4 = "#4C72B0", "#DD8452", "#C44E52", "#55A868"

metrics = {}
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=200)
    plt.close()


def evaluate(y_true, y_pred):
    rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    return {
        "test_accuracy": round(accuracy_score(y_true, y_pred), 4),
        "test_macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "per_class_f1": {c: round(rep[c]["f1-score"], 4) for c in CLASS_NAMES},
        "per_class_recall": {c: round(rep[c]["recall"], 4) for c in CLASS_NAMES},
    }


def plot_cm(ax, cm, title):
    cm_pct = 100 * cm / cm.sum(axis=1, keepdims=True)
    ax.grid(False)
    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(10))
    ax.set_xticklabels(SHORT, rotation=90, fontsize=8)
    ax.set_yticks(range(10))
    ax.set_yticklabels(SHORT, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(10):
        for j in range(10):
            if round(cm_pct[i, j]) >= 1:
                ax.text(j, i, f"{cm_pct[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm_pct[i, j] > 50 else "black")


def upper_share(cm):
    """Share of all misclassifications that are confusions among the upper-body classes."""
    off = cm - np.diag(np.diag(cm))
    return round(float(off[np.ix_(UPPER, UPPER)].sum() / off.sum()), 3)


def tree_diversity(estimators, X, y):
    """Mean accuracy of the individual trees and mean pairwise correlation of
    their correct/incorrect indicators (an estimate of rho in the slides)."""
    correct = np.array([est.predict(X) == y for est in estimators], dtype=float)
    corr = np.corrcoef(correct)
    iu = np.triu_indices_from(corr, k=1)
    return {"mean_tree_accuracy": round(float(correct.mean()), 4),
            "mean_pairwise_error_corr": round(float(np.nanmean(corr[iu])), 4)}


# =====================================================================
# Data
# =====================================================================
log("loading data")
# Load Fashion-MNIST from Keras (downloads on first run if not cached)
(x_trainset, y_trainset), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()


def square2row(square, num_obs):
    """Reshape a stack of 28x28 images into (num_obs, 784) flat vectors."""
    return np.resize(square, [num_obs, 28 * 28])


X_train_full, y_train_full = square2row(x_trainset, x_trainset.shape[0]), y_trainset
X_test, y_test = square2row(x_test, x_test.shape[0]), y_test
X_train_full = X_train_full.astype(np.float32) / 255.0   # scale to [0,1]
X_test = X_test.astype(np.float32) / 255.0

metrics["n_train"] = int(len(y_train_full))
metrics["n_test"] = int(len(y_test))
metrics["train_class_counts"] = np.bincount(y_train_full).tolist()
metrics["test_class_counts"] = np.bincount(y_test).tolist()

# ---------------------------------------------------------------------
# Data analysis
# ---------------------------------------------------------------------
log("data analysis")
frac_zero = (X_train_full == 0).mean(axis=0)
imgs = X_train_full.reshape(-1, 28, 28)
rng = np.random.default_rng(RNG)
sub = imgs[rng.choice(len(imgs), 10000, replace=False)]


def mean_corr(a, b):
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)
    keep = (a.std(axis=0) > 0.05) & (b.std(axis=0) > 0.05)
    a, b = a[:, keep], b[:, keep]
    a = (a - a.mean(0)) / a.std(0)
    b = (b - b.mean(0)) / b.std(0)
    return float((a * b).mean(0).mean())


# average image per class
class_means = np.stack([X_train_full[y_train_full == c].mean(axis=0) for c in range(10)])
mean_img_corr = np.corrcoef(class_means)

# baseline: classify each test image by the closest average image
dists = ((X_test ** 2).sum(1)[:, None] - 2 * X_test @ class_means.T
         + (class_means ** 2).sum(1)[None, :])
y_pred_proto = dists.argmin(axis=1)

# k-means: do the classes form separate clusters?
km = KMeans(n_clusters=10, n_init=5, random_state=RNG).fit(X_train_full)
cont = np.zeros((10, 10), dtype=int)          # rows: classes, cols: clusters
for c, k in zip(y_train_full, km.labels_):
    cont[c, k] += 1
cluster_major_class = cont.argmax(axis=0)
purity = float(cont.max(axis=0).sum() / cont.sum())
class_in_main_cluster = {CLASS_NAMES[c]: round(float(cont[c].max() / cont[c].sum()), 3)
                         for c in range(10)}
# share of upper-body images that end up in a cluster dominated by another upper-body class
upper_mixed = float(sum(cont[c, k] for c in UPPER for k in range(10)
                        if cluster_major_class[k] in UPPER and cluster_major_class[k] != c)
                    / sum(cont[c].sum() for c in UPPER))

metrics["data_analysis"] = {
    "pixels_zero_in_90pct_of_images": int(np.sum(frac_zero > 0.90)),
    "mean_corr_horizontal_neighbours": round(mean_corr(sub[:, :, :-1], sub[:, :, 1:]), 3),
    "mean_corr_pixels_7_apart": round(mean_corr(sub[:, :, :-7], sub[:, :, 7:]), 3),
    "class_mean_image_corr": np.round(mean_img_corr, 3).tolist(),
    "prototype_classifier": evaluate(y_test, y_pred_proto),
    "kmeans_contingency_class_by_cluster": cont.tolist(),
    "kmeans_cluster_majority_class": [CLASS_NAMES[c] for c in cluster_major_class],
    "kmeans_purity": round(purity, 3),
    "kmeans_class_share_in_main_cluster": class_in_main_cluster,
    "kmeans_upper_body_in_other_upper_body_cluster": round(upper_mixed, 3),
}
log(f"prototype acc {metrics['data_analysis']['prototype_classifier']['test_accuracy']}, "
    f"kmeans purity {purity:.3f}")

fig, axes = plt.subplots(2, 10, figsize=(12, 2.9))
for c in range(10):
    idx = np.where(y_train_full == c)[0]
    axes[0, c].imshow(X_train_full[idx[0]].reshape(28, 28), cmap="gray_r")
    axes[0, c].set_title(SHORT[c], fontsize=10)
    axes[1, c].imshow(class_means[c].reshape(28, 28), cmap="gray_r")
    for r in range(2):
        axes[r, c].grid(False)
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
axes[0, 0].set_ylabel("example", fontsize=10)
axes[1, 0].set_ylabel("class mean", fontsize=10)
savefig("data_overview.png")

# k-means figure: share of each class in each cluster, clusters ordered by majority class
order = np.argsort(cluster_major_class, kind="stable")
share = 100 * cont[:, order] / cont.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.grid(False)
ax.imshow(share, cmap="Blues", vmin=0, vmax=100)
ax.set_yticks(range(10))
ax.set_yticklabels(SHORT, fontsize=8)
ax.set_xticks(range(10))
ax.set_xticklabels([f"{i + 1} ({SHORT[cluster_major_class[k]]})" for i, k in enumerate(order)],
                   fontsize=7.5, rotation=90)
ax.set_xlabel("k-means cluster (majority class)")
ax.set_ylabel("true class")
ax.set_title("Share of each class per cluster (%)")
for i in range(10):
    for j in range(10):
        if round(share[i, j]) >= 1:
            ax.text(j, i, f"{share[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if share[i, j] > 50 else "black")
savefig("kmeans.png")

# =====================================================================
# Part 1a: decision tree
# =====================================================================
log("decision tree: 3-fold CV grid search over max_depth and min_samples_leaf")
depths = [6, 8, 10, 12, 14, 16, 20, None]
leaves = [1, 10, 30]
grid = GridSearchCV(DecisionTreeClassifier(random_state=RNG),
                    {"max_depth": depths, "min_samples_leaf": leaves},
                    cv=3, n_jobs=-1, return_train_score=True)
grid.fit(X_train_full, y_train_full)
res = grid.cv_results_


def cv_curve(leaf, key):
    out = []
    for d in depths:
        i = [k for k, p in enumerate(res["params"])
             if p["max_depth"] == d and p["min_samples_leaf"] == leaf][0]
        out.append(float(res[key][i]))
    return out


depth_labels = [str(d) if d is not None else "None" for d in depths]
metrics["tree_grid"] = {
    "depths": depth_labels, "min_samples_leaf": leaves,
    "cv_acc": {str(l): cv_curve(l, "mean_test_score") for l in leaves},
    "cv_std": {str(l): cv_curve(l, "std_test_score") for l in leaves},
    "train_acc": {str(l): cv_curve(l, "mean_train_score") for l in leaves},
    "best_params": {k: (v if v is not None else "None") for k, v in grid.best_params_.items()},
    "best_cv_acc": round(float(grid.best_score_), 4),
}
log(f"best tree params {grid.best_params_} cv={grid.best_score_:.4f}")

log("decision tree: cost-complexity pruning (3-fold CV)")
alphas = [0.0, 5e-5, 1e-4, 2e-4, 5e-4]
grid_ccp = GridSearchCV(DecisionTreeClassifier(random_state=RNG), {"ccp_alpha": alphas},
                        cv=3, n_jobs=-1)
grid_ccp.fit(X_train_full, y_train_full)
metrics["tree_ccp"] = {
    "alphas": alphas,
    "cv_acc": [float(s) for s in grid_ccp.cv_results_["mean_test_score"]],
    "best_alpha": grid_ccp.best_params_["ccp_alpha"],
    "best_cv_acc": round(float(grid_ccp.best_score_), 4),
    "best_n_leaves": int(grid_ccp.best_estimator_.get_n_leaves()),
}
log(f"ccp best {grid_ccp.best_params_} cv={grid_ccp.best_score_:.4f}")

# final tree: the best of the two searches by CV accuracy (both refitted on all 60,000 images)
if grid_ccp.best_score_ > grid.best_score_:
    tree, tree_params = grid_ccp.best_estimator_, {"ccp_alpha": grid_ccp.best_params_["ccp_alpha"]}
else:
    tree, tree_params = grid.best_estimator_, metrics["tree_grid"]["best_params"]
y_pred_tree = tree.predict(X_test)
root = int(tree.tree_.feature[0])
metrics["decision_tree"] = {
    "params": tree_params,
    "n_leaves": int(tree.get_n_leaves()),
    "depth": int(tree.get_depth()),
    "root_split_pixel_row_col": [root // 28, root % 28],
    "root_split_threshold": round(float(tree.tree_.threshold[0]), 3),
    **evaluate(y_test, y_pred_tree),
}
log(f"tree test acc {metrics['decision_tree']['test_accuracy']}")

# =====================================================================
# Part 1b: ensembles, tuned with 3-fold CV on the full training set
# (same idea as GridSearchCV for the tree, written out as a loop so that the
# forest can grow tree by tree with warm_start and the fitted trees of every
# fold can be inspected)
# =====================================================================
folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=RNG)
             .split(X_train_full, y_train_full))
n_trees_grid = [10, 25, 50, 100, 200]
m_grid = [7, 14, 28, 56, 112]
acc_trees = np.zeros((3, len(n_trees_grid)))
acc_m = np.zeros((3, len(m_grid)))
acc_bag = np.zeros(3)
div = {"bagging": [], "random_forest_m28": [], "random_forest_m7": []}
for f, (i_tr, i_va) in enumerate(folds):
    Xa, ya, Xb, yb = X_train_full[i_tr], y_train_full[i_tr], X_train_full[i_va], y_train_full[i_va]

    log(f"fold {f + 1}: random forest, number of trees")
    rf = RandomForestClassifier(n_estimators=10, max_features="sqrt", warm_start=True,
                                n_jobs=-1, random_state=RNG)
    for j, nt in enumerate(n_trees_grid):
        rf.set_params(n_estimators=nt)
        rf.fit(Xa, ya)                  # warm start: adds trees to the existing forest
        acc_trees[f, j] = accuracy_score(yb, rf.predict(Xb))

    log(f"fold {f + 1}: random forest, subset size m (50 trees)")
    for j, m in enumerate(m_grid):
        model = RandomForestClassifier(n_estimators=50, max_features=m, n_jobs=-1, random_state=RNG)
        model.fit(Xa, ya)
        acc_m[f, j] = accuracy_score(yb, model.predict(Xb))
        if m in (7, 28):
            div[f"random_forest_m{m}"].append(tree_diversity(model.estimators_, Xb, yb))

    log(f"fold {f + 1}: bagging (m = p = 784, 50 trees)")
    bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, n_jobs=-1, random_state=RNG)
    bag.fit(Xa, ya)
    acc_bag[f] = accuracy_score(yb, bag.predict(Xb))
    div["bagging"].append(tree_diversity(bag.estimators_, Xb, yb))
    log(f"  fold {f + 1}: trees {np.round(acc_trees[f], 4)}, m {np.round(acc_m[f], 4)}, bag {acc_bag[f]:.4f}")

rf_curve = acc_trees.mean(axis=0).tolist()
m_curve = acc_m.mean(axis=0).tolist()
bag_acc = float(acc_bag.mean())
metrics["rf_n_trees"] = {"n_trees": n_trees_grid, "cv_acc": rf_curve,
                         "cv_std": acc_trees.std(axis=0).tolist()}
metrics["rf_m"] = {
    "m": m_grid + [784], "cv_acc": m_curve + [bag_acc],
    "cv_std": acc_m.std(axis=0).tolist() + [float(acc_bag.std())],
    "best_m": m_grid[int(np.argmax(m_curve))],
}
for k, v in div.items():               # averaged over the three folds
    metrics["rf_m"][k] = {key: round(float(np.mean([d[key] for d in v])), 4) for key in v[0]}
log(f"ensemble CV: {metrics['rf_m']}")


log("final random forest: 200 trees, m = sqrt(784) = 28, all training data")
rf_final = RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=-1,
                                  random_state=RNG)
t = time.time()
rf_final.fit(X_train_full, y_train_full)
rf_time = time.time() - t
y_pred_rf = rf_final.predict(X_test)
metrics["random_forest"] = {
    "n_estimators": 200, "max_features": "sqrt (28)",
    "mean_tree_depth": round(float(np.mean([e.get_depth() for e in rf_final.estimators_])), 1),
    "mean_tree_leaves": int(np.mean([e.get_n_leaves() for e in rf_final.estimators_])),
    "fit_time_sec": round(rf_time, 1),
    **evaluate(y_test, y_pred_rf),
}
log(f"rf test acc {metrics['random_forest']['test_accuracy']}")

cm_tree = confusion_matrix(y_test, y_pred_tree)
cm_rf = confusion_matrix(y_test, y_pred_rf)
metrics["confusion"] = {
    "tree": cm_tree.tolist(), "random_forest": cm_rf.tolist(),
    "tree_errors_within_upper_body": upper_share(cm_tree),
    "rf_errors_within_upper_body": upper_share(cm_rf),
    "tree_right_rf_wrong": int(np.sum((y_pred_tree == y_test) & (y_pred_rf != y_test))),
    "tree_wrong_rf_right": int(np.sum((y_pred_tree != y_test) & (y_pred_rf == y_test))),
}

# ---------------------------------------------------------------------
# Robustness to noise; same noise as gradient_boosting.py
# ---------------------------------------------------------------------
log("noise robustness")
noise = {}
noise_rng = np.random.default_rng(RNG + 1)
for sd in [0.1, 0.2]:
    X_noisy = np.clip(X_test + noise_rng.normal(0, sd, X_test.shape).astype(np.float32), 0, 1)
    noise[str(sd)] = {
        "tree": round(accuracy_score(y_test, tree.predict(X_noisy)), 4),
        "random_forest": round(accuracy_score(y_test, rf_final.predict(X_noisy)), 4),
    }
metrics["noise_robustness_test_acc"] = noise
log(f"noise: {noise}")

# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
ax = axes[0]
for leaf, col in zip(leaves, [C1, C2, C4]):
    ax.plot(depth_labels, metrics["tree_grid"]["cv_acc"][str(leaf)], marker="o", ms=3.5,
            color=col, label=f"CV, min_samples_leaf={leaf}")
ax.plot(depth_labels, metrics["tree_grid"]["train_acc"]["1"], marker="o", ms=3.5, ls="--",
        color=C1, alpha=0.6, label="training, min_samples_leaf=1")
ax.set_xlabel("max_depth")
ax.set_ylabel("accuracy")
ax.set_title("(a) Decision tree (3-fold CV)")
ax.legend(fontsize=7.5)
ax = axes[1]
ax.plot(n_trees_grid, rf_curve, marker="o", ms=3.5, color=C3)
ax.set_xlabel("number of trees")
ax.set_ylabel("3-fold CV accuracy")
ax.set_title("(b) Random forest: number of trees")
ax = axes[2]
ax.plot(m_grid, m_curve, marker="o", ms=3.5, color=C3, label="random forest")
ax.plot([784], [bag_acc], marker="s", ms=5, color=C4, ls="none", label="bagging (m = p)")
ax.axvline(28, color="grey", ls=":", lw=1)
ax.set_xscale("log")
ax.set_xticks(m_grid + [784])
ax.set_xticklabels([str(m) for m in m_grid] + ["784"])
ax.minorticks_off()
ax.set_xlabel("m = pixels considered per split (log scale)")
ax.set_ylabel("3-fold CV accuracy")
ax.set_title("(c) Ensembles of 50 trees: subset size m")
ax.legend(loc="lower center")
savefig("tuning_curves.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
plot_cm(axes[0], cm_tree, f"(a) Decision tree (accuracy {metrics['decision_tree']['test_accuracy']:.1%})")
plot_cm(axes[1], cm_rf, f"(b) Random forest (accuracy {metrics['random_forest']['test_accuracy']:.1%})")
savefig("confusion_matrices.png")


np.save(os.path.join(RES_DIR, "rf_test_predictions.npy"), y_pred_rf)
with open(os.path.join(RES_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

log("done")
for name in ["decision_tree", "random_forest"]:
    print(f"  {name:15s} acc={metrics[name]['test_accuracy']:.4f} "
          f"macro-F1={metrics[name]['test_macro_f1']:.4f}")
