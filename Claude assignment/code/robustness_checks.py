"""
Two extra checks on the Part 1 tuning choices (validation split only):
  1. cost-complexity pruning (ccp_alpha) as an alternative to max_depth
  2. the number of candidate pixels per split in the random forest
     (max_features = 14, 28, 56; 50 trees each)

Run: python robustness_checks.py   (about 10 minutes on a laptop CPU)
Writes: ../results/robustness.json
"""
import json
import os

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_utils import load_fashion_mnist

RNG = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RES_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES_DIR, exist_ok=True)

X, y, _, _ = load_fashion_mnist(DATA_DIR)
X = X.astype(np.float32) / 255.0
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.10, stratify=y, random_state=RNG)
out = {}

# 1. cost-complexity pruning of a fully grown tree
print("Cost-complexity pruning...")
alphas = [0.0, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
ccp = []
for a in alphas:
    t = DecisionTreeClassifier(ccp_alpha=a, random_state=RNG).fit(X_tr, y_tr)
    acc = accuracy_score(y_val, t.predict(X_val))
    ccp.append({"ccp_alpha": a, "val_acc": round(acc, 4),
                "n_leaves": int(t.get_n_leaves()), "depth": int(t.get_depth())})
    print(" ", ccp[-1])
out["ccp_alpha"] = ccp

# root split of the chosen depth-12 tree (interpretability)
t12 = DecisionTreeClassifier(max_depth=12, random_state=RNG).fit(X_tr, y_tr)
root = int(t12.tree_.feature[0])
out["depth12_root_split"] = {"pixel_row": root // 28, "pixel_col": root % 28,
                             "threshold": round(float(t12.tree_.threshold[0]), 3)}
print("root split:", out["depth12_root_split"])

# 2. number of candidate pixels per split in the forest
print("Random forest max_features...")
mf = []
for m in [14, 28, 56]:
    rf = RandomForestClassifier(n_estimators=50, max_features=m, n_jobs=-1, random_state=RNG)
    rf.fit(X_tr, y_tr)
    acc = accuracy_score(y_val, rf.predict(X_val))
    mf.append({"max_features": m, "val_acc": round(acc, 4)})
    print(" ", mf[-1])
out["rf_max_features_50_trees"] = mf

with open(os.path.join(RES_DIR, "robustness.json"), "w") as f:
    json.dump(out, f, indent=2)
print("Done.")
