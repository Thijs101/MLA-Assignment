"""
MLA final project - Part 1: decision tree and tree ensembles on Fashion-MNIST.

Runs the full Part 1 analysis and writes all numbers to ../results/part1.json and
all figures of the report (except the CNN curves) to ../figures/.
Running everything takes about 3 hours on a laptop; most of that is the
gradient boosting model near the end.

Usage (from the code/ folder):  python part1_trees.py
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
from tensorflow import keras
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              GradientBoostingClassifier)
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

SEED = 42
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
SHORT = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
         "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
UPPER = [0, 2, 4, 6]   # T-shirt/top, Pullover, Coat, Shirt

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.bbox": "tight"})
BLUE, ORANGE, RED, GREEN = "#4C72B0", "#DD8452", "#C44E52", "#55A868"

results = {}


def save_figure(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=200)
    plt.close()


def scores(y_true, y_pred):
    rep = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    return {"test_accuracy": round(accuracy_score(y_true, y_pred), 4),
            "test_macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
            "per_class_f1": {c: round(rep[c]["f1-score"], 4) for c in CLASS_NAMES}}


def plot_confusion(ax, cm, title):
    pct = 100 * cm / cm.sum(axis=1, keepdims=True)
    ax.grid(False)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(10))
    ax.set_xticklabels(SHORT, rotation=90, fontsize=8)
    ax.set_yticks(range(10))
    ax.set_yticklabels(SHORT, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(10):
        for j in range(10):
            if round(pct[i, j]) >= 1:
                ax.text(j, i, f"{pct[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if pct[i, j] > 50 else "black")


def upper_body_share(cm):
    # share of all mistakes that are confusions between T-shirt, Pullover, Coat and Shirt
    wrong = cm - np.diag(np.diag(cm))
    return round(float(wrong[np.ix_(UPPER, UPPER)].sum() / wrong.sum()), 3)


def tree_diversity(trees, X, y):
    # average accuracy of the single trees and average correlation of their errors
    correct = np.array([t.predict(X) == y for t in trees], dtype=float)
    corr = np.corrcoef(correct)[np.triu_indices(len(trees), k=1)]
    return {"mean_tree_accuracy": round(float(correct.mean()), 4),
            "mean_pairwise_error_corr": round(float(np.nanmean(corr)), 4)}


def shap_chunk(model, X):
    sv = np.asarray(shap.TreeExplainer(model).shap_values(X, check_additivity=False))
    if sv.shape[-1] == 10:
        sv = np.moveaxis(sv, -1, 0)
    return sv


def shap_values(model, X):
    # SHAP values (classes x images x pixels), computed in chunks on all CPU cores
    parts = Parallel(n_jobs=-1)(delayed(shap_chunk)(model, X[i:i + 25]) for i in range(0, len(X), 25))
    return np.concatenate(parts, axis=1)


def first_per_class(y, n):
    return np.concatenate([np.where(y == c)[0][:n] for c in range(10)])


def square2row(square, num_obs):
    return np.resize(square, [num_obs, 28 * 28])


# Load the data
(x_trainset, y_trainset), (x_testset, y_testset) = keras.datasets.fashion_mnist.load_data()
X_train = square2row(x_trainset, x_trainset.shape[0]).astype(np.float32) / 255.0
y_train = y_trainset
X_test = square2row(x_testset, x_testset.shape[0]).astype(np.float32) / 255.0
y_test = y_testset

# Data analysis
class_means = np.stack([X_train[y_train == c].mean(axis=0) for c in range(10)])

# baseline: label each test image with the class of the closest average image
dist = ((X_test ** 2).sum(1)[:, None] - 2 * X_test @ class_means.T
        + (class_means ** 2).sum(1)[None, :])
y_pred_avg = dist.argmin(axis=1)

# correlation between neighbouring pixels (on a random sample of 10,000 images)
imgs = X_train.reshape(-1, 28, 28)
sample = imgs[np.random.default_rng(SEED).choice(len(imgs), 10000, replace=False)]


def pixel_corr(a, b):
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    keep = (a.std(axis=0) > 0.05) & (b.std(axis=0) > 0.05)
    a = (a[:, keep] - a[:, keep].mean(0)) / a[:, keep].std(0)
    b = (b[:, keep] - b[:, keep].mean(0)) / b[:, keep].std(0)
    return float((a * b).mean(0).mean())


# k-means with ten clusters: which classes end up together?
km = KMeans(n_clusters=10, n_init=5, random_state=SEED).fit(X_train)
cont = np.zeros((10, 10), dtype=int)
for c, k in zip(y_train, km.labels_):
    cont[c, k] += 1
majority = cont.argmax(axis=0)
mixed = sum(cont[c, k] for c in UPPER for k in range(10) if majority[k] in UPPER and majority[k] != c)

results["data_analysis"] = {
    "pixels_zero_in_90pct_of_images": int(np.sum((X_train == 0).mean(axis=0) > 0.90)),
    "mean_corr_horizontal_neighbours": round(pixel_corr(sample[:, :, :-1], sample[:, :, 1:]), 3),
    "mean_corr_pixels_7_apart": round(pixel_corr(sample[:, :, :-7], sample[:, :, 7:]), 3),
    "class_mean_image_corr": np.round(np.corrcoef(class_means), 3).tolist(),
    "prototype_classifier": scores(y_test, y_pred_avg),
    "kmeans_contingency_class_by_cluster": cont.tolist(),
    "kmeans_class_share_in_main_cluster": {CLASS_NAMES[c]: round(float(cont[c].max() / cont[c].sum()), 3)
                                           for c in range(10)},
    "kmeans_upper_body_in_other_upper_body_cluster": round(float(mixed / cont[UPPER].sum()), 3),
}
print("Average-image baseline, test accuracy:", results["data_analysis"]["prototype_classifier"]["test_accuracy"])

fig, axes = plt.subplots(2, 10, figsize=(12, 2.9))
for c in range(10):
    axes[0, c].imshow(X_train[np.where(y_train == c)[0][0]].reshape(28, 28), cmap="gray_r")
    axes[0, c].set_title(SHORT[c], fontsize=10)
    axes[1, c].imshow(class_means[c].reshape(28, 28), cmap="gray_r")
    for r in range(2):
        axes[r, c].grid(False)
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
axes[0, 0].set_ylabel("example", fontsize=10)
axes[1, 0].set_ylabel("class mean", fontsize=10)
save_figure("data_overview.png")

order = np.argsort(majority, kind="stable")
share = 100 * cont[:, order] / cont.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.grid(False)
ax.imshow(share, cmap="Blues", vmin=0, vmax=100)
ax.set_yticks(range(10))
ax.set_yticklabels(SHORT, fontsize=8)
ax.set_xticks(range(10))
ax.set_xticklabels([f"{i + 1} ({SHORT[majority[k]]})" for i, k in enumerate(order)], fontsize=7.5, rotation=90)
ax.set_xlabel("k-means cluster (majority class)")
ax.set_ylabel("true class")
ax.set_title("Share of each class per cluster (%)")
for i in range(10):
    for j in range(10):
        if round(share[i, j]) >= 1:
            ax.text(j, i, f"{share[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if share[i, j] > 50 else "black")
save_figure("kmeans.png")

# Part 1a: decision tree
depths = [6, 8, 10, 12, 14, 16, 20, None]
leaf_sizes = [1, 10, 30]
grid = GridSearchCV(DecisionTreeClassifier(random_state=SEED),
                    {"max_depth": depths, "min_samples_leaf": leaf_sizes},
                    cv=3, n_jobs=-1, return_train_score=True)
grid.fit(X_train, y_train)


def curve(leaf, key):
    out = []
    for d in depths:
        i = [k for k, p in enumerate(grid.cv_results_["params"])
             if p["max_depth"] == d and p["min_samples_leaf"] == leaf][0]
        out.append(float(grid.cv_results_[key][i]))
    return out


depth_labels = ["None" if d is None else str(d) for d in depths]
results["tree_grid"] = {
    "depths": depth_labels, "min_samples_leaf": leaf_sizes,
    "cv_acc": {str(l): curve(l, "mean_test_score") for l in leaf_sizes},
    "train_acc": {str(l): curve(l, "mean_train_score") for l in leaf_sizes},
    "best_params": {k: ("None" if v is None else v) for k, v in grid.best_params_.items()},
    "best_cv_acc": round(float(grid.best_score_), 4),
}

alphas = [0.0, 5e-5, 1e-4, 2e-4, 5e-4]
grid_ccp = GridSearchCV(DecisionTreeClassifier(random_state=SEED), {"ccp_alpha": alphas}, cv=3, n_jobs=-1)
grid_ccp.fit(X_train, y_train)
results["tree_ccp"] = {"alphas": alphas,
                       "cv_acc": [float(s) for s in grid_ccp.cv_results_["mean_test_score"]],
                       "best_alpha": grid_ccp.best_params_["ccp_alpha"],
                       "best_cv_acc": round(float(grid_ccp.best_score_), 4)}

# final tree: whichever search had the best CV accuracy (refitted on all training images)
best = grid_ccp if grid_ccp.best_score_ > grid.best_score_ else grid
tree = best.best_estimator_
y_pred_tree = tree.predict(X_test)
results["decision_tree"] = {
    "params": {k: ("None" if v is None else v) for k, v in best.best_params_.items()},
    "n_leaves": int(tree.get_n_leaves()),
    "depth": int(tree.get_depth()),
    "root_split_pixel_row_col": [int(tree.tree_.feature[0]) // 28, int(tree.tree_.feature[0]) % 28],
    **scores(y_test, y_pred_tree),
}
print("Decision tree", results["decision_tree"]["params"], "test accuracy:",
      results["decision_tree"]["test_accuracy"])

# Part 1b: random forest and bagging
folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(X_train, y_train))
n_trees = [10, 25, 50, 100, 200]
m_values = [7, 14, 28, 56, 112]
acc_trees = np.zeros((3, len(n_trees)))
acc_m = np.zeros((3, len(m_values)))
acc_bag = np.zeros(3)
diversity = {"bagging": [], "random_forest_m28": [], "random_forest_m7": []}

for f, (i_tr, i_va) in enumerate(folds):
    Xa, ya, Xb, yb = X_train[i_tr], y_train[i_tr], X_train[i_va], y_train[i_va]

    # number of trees (warm_start adds trees to the same forest)
    rf = RandomForestClassifier(n_estimators=10, max_features="sqrt", warm_start=True,
                                n_jobs=-1, random_state=SEED)
    for j, n in enumerate(n_trees):
        rf.set_params(n_estimators=n)
        rf.fit(Xa, ya)
        acc_trees[f, j] = accuracy_score(yb, rf.predict(Xb))

    # number of pixels m considered at each split, 50 trees each
    for j, m in enumerate(m_values):
        rf = RandomForestClassifier(n_estimators=50, max_features=m, n_jobs=-1, random_state=SEED)
        rf.fit(Xa, ya)
        acc_m[f, j] = accuracy_score(yb, rf.predict(Xb))
        if m in (7, 28):
            diversity[f"random_forest_m{m}"].append(tree_diversity(rf.estimators_, Xb, yb))

    # plain bagging (all 784 pixels at every split), 50 trees
    bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, n_jobs=-1, random_state=SEED)
    bag.fit(Xa, ya)
    acc_bag[f] = accuracy_score(yb, bag.predict(Xb))
    diversity["bagging"].append(tree_diversity(bag.estimators_, Xb, yb))
    print(f"Fold {f + 1} of 3 done")

results["rf_n_trees"] = {"n_trees": n_trees, "cv_acc": acc_trees.mean(axis=0).tolist(),
                         "cv_std": acc_trees.std(axis=0).tolist()}
results["rf_m"] = {"m": m_values + [784],
                   "cv_acc": acc_m.mean(axis=0).tolist() + [float(acc_bag.mean())],
                   "cv_std": acc_m.std(axis=0).tolist() + [float(acc_bag.std())]}
for name, folds_div in diversity.items():
    results["rf_m"][name] = {k: round(float(np.mean([d[k] for d in folds_div])), 4) for k in folds_div[0]}

# final forest: 200 trees, m = sqrt(784) = 28, all training images
timing = {"decision_tree": {"fit_sec": round(float(best.refit_time_), 1)}}
forest = RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=SEED)
start = time.time()
forest.fit(X_train, y_train)
timing["random_forest"] = {"fit_sec": round(time.time() - start, 1)}
y_pred_rf = forest.predict(X_test)
results["random_forest"] = {
    "mean_tree_depth": round(float(np.mean([t.get_depth() for t in forest.estimators_])), 1),
    "mean_tree_leaves": int(np.mean([t.get_n_leaves() for t in forest.estimators_])),
    **scores(y_test, y_pred_rf),
}
print("Random forest test accuracy:", results["random_forest"]["test_accuracy"])

# same forest with m = 112, only to compare the training time
start = time.time()
RandomForestClassifier(n_estimators=200, max_features=112, n_jobs=-1, random_state=SEED).fit(X_train, y_train)
timing["random_forest_m112"] = {"fit_sec": round(time.time() - start, 1)}

cm_tree = confusion_matrix(y_test, y_pred_tree)
cm_rf = confusion_matrix(y_test, y_pred_rf)
results["confusion"] = {
    "tree": cm_tree.tolist(), "random_forest": cm_rf.tolist(),
    "tree_errors_within_upper_body": upper_body_share(cm_tree),
    "rf_errors_within_upper_body": upper_body_share(cm_rf),
    "tree_right_rf_wrong": int(np.sum((y_pred_tree == y_test) & (y_pred_rf != y_test))),
    "tree_wrong_rf_right": int(np.sum((y_pred_tree != y_test) & (y_pred_rf == y_test))),
}

# robustness: Gaussian noise added to the test images
noise_rng = np.random.default_rng(SEED + 1)
noisy_tests = {sd: np.clip(X_test + noise_rng.normal(0, sd, X_test.shape).astype(np.float32), 0, 1)
               for sd in [0.1, 0.2]}
results["noise_robustness_test_acc"] = {
    str(sd): {"tree": round(accuracy_score(y_test, tree.predict(X)), 4),
              "random_forest": round(accuracy_score(y_test, forest.predict(X)), 4)}
    for sd, X in noisy_tests.items()}

# SHAP values of the final forest on 1000 test images (100 per class)
idx = first_per_class(y_test, 100)
sv = shap_values(forest, X_test[idx])
mean_abs_shap = np.abs(sv).mean(axis=(0, 1))
shirt_shap = sv[6][y_test[idx] == 6].mean(axis=0)
results["shap"] = {"final_forest": {
    "n_images": int(len(idx)),
    "corr_shap_vs_impurity": round(float(np.corrcoef(mean_abs_shap, forest.feature_importances_)[0, 1]), 3),
    "top20_pixels_row_col": [[int(p) // 28, int(p) % 28] for p in np.argsort(mean_abs_shap)[::-1][:20]],
}}

# pixel selection on the first CV fold: rank the pixels by SHAP on 1,000 held-out images,
# then score forests on only the top-k pixels on the other held-out images
i_tr, i_va = folds[0]
Xa, ya, Xb, yb = X_train[i_tr], y_train[i_tr], X_train[i_va], y_train[i_va]
rank_idx = first_per_class(yb, 100)
score = np.ones(len(yb), dtype=bool)
score[rank_idx] = False
rf = RandomForestClassifier(n_estimators=50, max_features="sqrt", n_jobs=-1, random_state=SEED).fit(Xa, ya)
ranking = np.argsort(np.abs(shap_values(rf, Xb[rank_idx])).mean(axis=(0, 1)))[::-1]
selection = {"784": round(accuracy_score(yb[score], rf.predict(Xb[score])), 4)}
for k in [50, 100, 200, 400]:
    rf_k = RandomForestClassifier(n_estimators=50, max_features="sqrt", n_jobs=-1,
                                  random_state=SEED).fit(Xa[:, ranking[:k]], ya)
    selection[str(k)] = round(accuracy_score(yb[score], rf_k.predict(Xb[score][:, ranking[:k]])), 4)
results["shap"]["pixel_selection"] = {"accuracy": selection}

# Gradient boosting with default settings
gb = GradientBoostingClassifier(random_state=SEED)
start = time.time()
gb.fit(X_train, y_train)
timing["gradient_boosting"] = {"fit_sec": round(time.time() - start, 1)}
timing["n_cpu_cores"] = os.cpu_count()
y_pred_gb = gb.predict(X_test)
gb_test = [accuracy_score(y_test, p) for p in gb.staged_predict(X_test)]
gb_train = [accuracy_score(y_train, p) for p in gb.staged_predict(X_train)]
results["gradient_boosting"] = {
    **scores(y_test, y_pred_gb),
    "fit_time_sec": timing["gradient_boosting"]["fit_sec"],
    "test_acc_per_round": [round(a, 4) for a in gb_test],
    "train_acc_per_round": [round(a, 4) for a in gb_train],
    "noise_robustness_test_acc": {str(sd): round(accuracy_score(y_test, gb.predict(X)), 4)
                                  for sd, X in noisy_tests.items()},
    "timing": timing,
}
print("Gradient boosting test accuracy:", results["gradient_boosting"]["test_accuracy"])

# XGBoost with default settings
xgb = XGBClassifier(n_jobs=-1, random_state=SEED)
start = time.time()
xgb.fit(X_train, y_train)
timing["xgboost"] = {"fit_sec": round(time.time() - start, 1)}
results["xgboost"] = {
    **scores(y_test, xgb.predict(X_test)),
    "train_accuracy": round(accuracy_score(y_train, xgb.predict(X_train)), 4),
    "noise_robustness_test_acc": {str(sd): round(accuracy_score(y_test, xgb.predict(X)), 4)
                                  for sd, X in noisy_tests.items()},
}
print("XGBoost test accuracy:", results["xgboost"]["test_accuracy"])

# Figures
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for leaf, col in zip(leaf_sizes, [BLUE, ORANGE, GREEN]):
    axes[0].plot(depth_labels, results["tree_grid"]["cv_acc"][str(leaf)], marker="o", ms=3.5,
                 color=col, label=f"CV, min_samples_leaf={leaf}")
axes[0].plot(depth_labels, results["tree_grid"]["train_acc"]["1"], marker="o", ms=3.5, ls="--",
             color=BLUE, alpha=0.6, label="training, min_samples_leaf=1")
axes[0].set_xlabel("max_depth")
axes[0].set_ylabel("accuracy")
axes[0].set_title("(a) Decision tree (3-fold CV)")
axes[0].legend(fontsize=7.5)
axes[1].plot(n_trees, results["rf_n_trees"]["cv_acc"], marker="o", ms=3.5, color=RED)
axes[1].set_xlabel("number of trees")
axes[1].set_ylabel("3-fold CV accuracy")
axes[1].set_title("(b) Random forest: number of trees")
axes[2].plot(m_values, acc_m.mean(axis=0), marker="o", ms=3.5, color=RED, label="random forest")
axes[2].plot([784], [acc_bag.mean()], marker="s", ms=5, color=GREEN, ls="none", label="bagging (m = p)")
axes[2].axvline(28, color="grey", ls=":", lw=1)
axes[2].set_xscale("log")
axes[2].set_xticks(m_values + [784])
axes[2].set_xticklabels([str(m) for m in m_values] + ["784"])
axes[2].minorticks_off()
axes[2].set_xlabel("m = pixels considered per split (log scale)")
axes[2].set_ylabel("3-fold CV accuracy")
axes[2].set_title("(c) Ensembles of 50 trees: subset size m")
axes[2].legend(loc="lower center")
save_figure("tuning_curves.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.9))
plot_confusion(axes[0], cm_tree, f"(a) Decision tree (accuracy {results['decision_tree']['test_accuracy']:.1%})")
plot_confusion(axes[1], cm_rf, f"(b) Random forest (accuracy {results['random_forest']['test_accuracy']:.1%})")
save_figure("confusion_matrices.png")

fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
panels = [(forest.feature_importances_, "inferno", False, "(a) Impurity decrease"),
          (mean_abs_shap, "inferno", False, "(b) Mean |SHAP|, all classes"),
          (shirt_shap, "RdBu_r", True, "(c) SHAP for 'Shirt' on shirts"),
          (np.abs(class_means[6] - class_means[2]), "inferno", False, "(d) |mean Shirt $-$ mean Pullover|")]
for ax, (img, cmap, symmetric, title) in zip(axes, panels):
    ax.grid(False)
    v = np.abs(img).max() if symmetric else None
    im = ax.imshow(img.reshape(28, 28), cmap=cmap, vmin=-v if symmetric else None, vmax=v)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046).ax.tick_params(labelsize=7)
save_figure("importance.png")

fig, ax = plt.subplots(figsize=(4.6, 3.4))
rounds = np.arange(1, len(gb_test) + 1)
ax.plot(rounds, gb_train, color=BLUE, label="training")
ax.plot(rounds, gb_test, color=RED, label="test")
ax.set_xlabel("boosting round")
ax.set_ylabel("accuracy")
ax.set_title("Gradient boosting: accuracy per round")
ax.legend(loc="lower right")
save_figure("gb_rounds.png")

# Summary of the test results
noise = results["noise_robustness_test_acc"]
summary = [("Decision tree", results["decision_tree"], noise["0.1"]["tree"], timing["decision_tree"]),
           ("Random forest", results["random_forest"], noise["0.1"]["random_forest"], timing["random_forest"]),
           ("Gradient boosting", results["gradient_boosting"],
            results["gradient_boosting"]["noise_robustness_test_acc"]["0.1"], timing["gradient_boosting"]),
           ("XGBoost", results["xgboost"], results["xgboost"]["noise_robustness_test_acc"]["0.1"], timing["xgboost"])]
print("\nModel              accuracy  average F1  Shirt F1  with noise  training time (s)")
for name, r, acc_noise, t in summary:
    print(f"{name:<18} {r['test_accuracy']:8.3f}  {r['test_macro_f1']:10.3f}  {r['per_class_f1']['Shirt']:8.2f}"
          f"  {acc_noise:10.3f}  {t['fit_sec']:17.1f}")

with open(os.path.join(RES_DIR, "part1.json"), "w") as f:
    json.dump(results, f, indent=2)
