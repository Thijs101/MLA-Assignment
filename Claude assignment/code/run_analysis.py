"""
Part 1: decision tree and tree-ensemble classification on Fashion-MNIST.

Produces:
  - results/metrics.json        summary numbers used in the report
  - figures/*.png                plots used in the report

Run: python run_analysis.py   (from the code/ directory, with ../data/
containing the four Fashion-MNIST idx.gz files)
"""
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)

from data_utils import load_fashion_mnist, CLASS_NAMES

RNG = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RES_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

metrics = {}


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150)
    plt.close()


print("Loading data...")
X_train_full, y_train_full, X_test, y_test = load_fashion_mnist(DATA_DIR)
print(f"train: {X_train_full.shape}, test: {X_test.shape}")

# scale pixel values to [0,1]
X_train_full = X_train_full.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

metrics["n_train"] = int(X_train_full.shape[0])
metrics["n_test"] = int(X_test.shape[0])
metrics["n_features"] = int(X_train_full.shape[1])
metrics["class_names"] = CLASS_NAMES
metrics["train_class_counts"] = np.bincount(y_train_full).tolist()
metrics["test_class_counts"] = np.bincount(y_test).tolist()

# ---------------------------------------------------------------
# Data exploration figures
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 10, figsize=(14, 3.2))
for c in range(10):
    idx = np.where(y_train_full == c)[0]
    for row in range(2):
        img = X_train_full[idx[row]].reshape(28, 28)
        ax = axes[row, c]
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        if row == 0:
            ax.set_title(CLASS_NAMES[c], fontsize=8)
savefig("sample_images.png")

plt.figure(figsize=(6, 3))
plt.bar(range(10), metrics["train_class_counts"], color="#4c72b0")
plt.xticks(range(10), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
plt.ylabel("training examples")
plt.title("Class balance in the training set")
savefig("class_balance.png")

# ---------------------------------------------------------------
# Train/validation split for model selection (test set stays untouched)
# ---------------------------------------------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.10, stratify=y_train_full, random_state=RNG
)
print(f"model-selection split -> train {X_tr.shape[0]}, val {X_val.shape[0]}")

# ---------------------------------------------------------------
# Part 1a: single decision tree
# ---------------------------------------------------------------
print("Tuning decision tree depth...")
depths = [4, 6, 8, 10, 12, 15, 20, 25, None]
train_acc, val_acc = [], []
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=RNG)
    clf.fit(X_tr, y_tr)
    train_acc.append(accuracy_score(y_tr, clf.predict(X_tr)))
    val_acc.append(accuracy_score(y_val, clf.predict(X_val)))
    print(f"  depth={d}: train={train_acc[-1]:.4f} val={val_acc[-1]:.4f}")

depth_labels = [str(d) if d is not None else "None" for d in depths]
plt.figure(figsize=(5.5, 4))
plt.plot(depth_labels, train_acc, marker="o", label="training accuracy")
plt.plot(depth_labels, val_acc, marker="o", label="validation accuracy")
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.title("Decision tree: effect of depth on over-fitting")
plt.legend()
savefig("tree_depth_curve.png")

best_depth_idx = int(np.argmax(val_acc))
best_depth = depths[best_depth_idx]
metrics["tree_depth_search"] = {
    "depths": depth_labels, "train_acc": train_acc, "val_acc": val_acc,
    "chosen_depth": depth_labels[best_depth_idx],
}
print(f"Chosen tree depth: {best_depth}")

t0 = time.time()
tree_final = DecisionTreeClassifier(max_depth=best_depth, random_state=RNG)
tree_final.fit(X_train_full, y_train_full)
tree_fit_time = time.time() - t0

y_pred_tree = tree_final.predict(X_test)
tree_test_acc = accuracy_score(y_test, y_pred_tree)
tree_test_f1 = f1_score(y_test, y_pred_tree, average="macro")
tree_report = classification_report(y_test, y_pred_tree, target_names=CLASS_NAMES, output_dict=True)

metrics["decision_tree"] = {
    "chosen_max_depth": best_depth if best_depth is not None else "None",
    "n_leaves": int(tree_final.get_n_leaves()),
    "fit_time_sec": round(tree_fit_time, 2),
    "test_accuracy": round(tree_test_acc, 4),
    "test_macro_f1": round(tree_test_f1, 4),
    "per_class_f1": {CLASS_NAMES[i]: round(tree_report[CLASS_NAMES[i]]["f1-score"], 4) for i in range(10)},
}
print("Decision tree test accuracy:", tree_test_acc)

cm = confusion_matrix(y_test, y_pred_tree)
plt.figure(figsize=(5.5, 5))
plt.imshow(cm, cmap="Blues")
plt.colorbar(fraction=0.046)
plt.xticks(range(10), CLASS_NAMES, rotation=90, fontsize=7)
plt.yticks(range(10), CLASS_NAMES, fontsize=7)
plt.xlabel("predicted")
plt.ylabel("true")
plt.title(f"Decision tree confusion matrix (acc={tree_test_acc:.3f})")
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  fontsize=6, color="white" if cm[i, j] > cm.max() / 2 else "black")
savefig("confusion_tree.png")

# ---------------------------------------------------------------
# Part 1b: random forest ensemble
# ---------------------------------------------------------------
print("Tuning random forest n_estimators...")
n_trees_grid = [10, 25, 50, 100, 200]
rf_val_acc = []
for nt in n_trees_grid:
    rf = RandomForestClassifier(n_estimators=nt, max_depth=None, n_jobs=-1, random_state=RNG)
    rf.fit(X_tr, y_tr)
    a = accuracy_score(y_val, rf.predict(X_val))
    rf_val_acc.append(a)
    print(f"  n_estimators={nt}: val={a:.4f}")

best_nt = n_trees_grid[int(np.argmax(rf_val_acc))]
metrics["rf_estimator_search"] = {"n_estimators": n_trees_grid, "val_acc": rf_val_acc, "chosen": best_nt}

plt.figure(figsize=(5.5, 4))
plt.plot(n_trees_grid, rf_val_acc, marker="o", color="#c44e52")
plt.xlabel("number of trees")
plt.ylabel("validation accuracy")
plt.title("Random forest: effect of ensemble size")
savefig("rf_ntrees_curve.png")

print(f"Chosen n_estimators: {best_nt}")
t0 = time.time()
rf_final = RandomForestClassifier(n_estimators=best_nt, oob_score=True, n_jobs=-1, random_state=RNG)
rf_final.fit(X_train_full, y_train_full)
rf_fit_time = time.time() - t0

y_pred_rf = rf_final.predict(X_test)
rf_test_acc = accuracy_score(y_test, y_pred_rf)
rf_test_f1 = f1_score(y_test, y_pred_rf, average="macro")
rf_report = classification_report(y_test, y_pred_rf, target_names=CLASS_NAMES, output_dict=True)

metrics["random_forest"] = {
    "chosen_n_estimators": best_nt,
    "fit_time_sec": round(rf_fit_time, 2),
    "test_accuracy": round(rf_test_acc, 4),
    "test_macro_f1": round(rf_test_f1, 4),
    "oob_accuracy": round(float(rf_final.oob_score_), 4),
    "per_class_f1": {CLASS_NAMES[i]: round(rf_report[CLASS_NAMES[i]]["f1-score"], 4) for i in range(10)},
}
print("Random forest test accuracy:", rf_test_acc, "OOB accuracy:", rf_final.oob_score_)

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5.5, 5))
plt.imshow(cm_rf, cmap="Blues")
plt.colorbar(fraction=0.046)
plt.xticks(range(10), CLASS_NAMES, rotation=90, fontsize=7)
plt.yticks(range(10), CLASS_NAMES, fontsize=7)
plt.xlabel("predicted")
plt.ylabel("true")
plt.title(f"Random forest confusion matrix (acc={rf_test_acc:.3f})")
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm_rf[i, j], ha="center", va="center",
                  fontsize=6, color="white" if cm_rf[i, j] > cm_rf.max() / 2 else "black")
savefig("confusion_rf.png")

# feature importance heat map
importances = rf_final.feature_importances_.reshape(28, 28)
plt.figure(figsize=(4, 4))
plt.imshow(importances, cmap="inferno")
plt.colorbar(fraction=0.046)
plt.title("Random forest pixel importance")
plt.axis("off")
savefig("rf_feature_importance.png")

# ---------------------------------------------------------------
# Secondary comparison: histogram-based gradient boosting
# (used only to support the discussion in 1b, not the chosen model)
# ---------------------------------------------------------------
print("Fitting gradient boosting comparison model...")
t0 = time.time()
gb = HistGradientBoostingClassifier(random_state=RNG)
gb.fit(X_train_full, y_train_full)
gb_fit_time = time.time() - t0
y_pred_gb = gb.predict(X_test)
gb_test_acc = accuracy_score(y_test, y_pred_gb)
gb_test_f1 = f1_score(y_test, y_pred_gb, average="macro")
gb_report = classification_report(y_test, y_pred_gb, target_names=CLASS_NAMES, output_dict=True)

metrics["gradient_boosting_comparison"] = {
    "per_class_f1": {CLASS_NAMES[i]: round(gb_report[CLASS_NAMES[i]]["f1-score"], 4) for i in range(10)},
    "model": "HistGradientBoostingClassifier (default settings)",
    "fit_time_sec": round(gb_fit_time, 2),
    "test_accuracy": round(gb_test_acc, 4),
    "test_macro_f1": round(gb_test_f1, 4),
}
print("Gradient boosting test accuracy:", gb_test_acc)

# ---------------------------------------------------------------
# Standard errors of the test accuracies (binomial approximation),
# to judge whether the differences between models are larger than
# what sampling noise on a 10,000-image test set would produce.
# ---------------------------------------------------------------
n_test = X_test.shape[0]
for name, acc in [("decision_tree", tree_test_acc), ("random_forest", rf_test_acc),
                   ("gradient_boosting_comparison", gb_test_acc)]:
    se = float(np.sqrt(acc * (1 - acc) / n_test))
    metrics[name]["test_accuracy_se"] = round(se, 4)

# ---------------------------------------------------------------
# Save everything
# ---------------------------------------------------------------
with open(os.path.join(RES_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\nDone. Summary:")
print(f"  Decision tree : acc={tree_test_acc:.4f}  macro-F1={tree_test_f1:.4f}  fit={tree_fit_time:.1f}s")
print(f"  Random forest : acc={rf_test_acc:.4f}  macro-F1={rf_test_f1:.4f}  fit={rf_fit_time:.1f}s")
print(f"  Grad. boosting: acc={gb_test_acc:.4f}  macro-F1={gb_test_f1:.4f}  fit={gb_fit_time:.1f}s")
