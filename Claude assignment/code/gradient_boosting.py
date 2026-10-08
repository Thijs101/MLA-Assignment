"""
MLA final project: gradient boosting reference model (Part 1b).

GradientBoostingClassifier with its default settings (100 rounds, learning
rate 0.1, trees of depth 3; ten trees per round, one per class), fitted on
all 60,000 training images. This takes a long time (about 80 minutes on a
laptop CPU), which is why it is a separate script from run_analysis.py.

It also measures the training and prediction time of the final decision tree
and random forest (and of a forest with m = 112 instead of 28) on the same machine (run run_analysis.py first: the tree
settings are read from ../results/metrics.json).

Run from the code/ folder:   python gradient_boosting.py
Writes: ../results/gradient_boosting.json
"""
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from tensorflow import keras

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

RNG = 42
HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "..", "results")
os.makedirs(RES_DIR, exist_ok=True)

# Load Fashion-MNIST from Keras (downloads on first run if not cached)
(x_trainset, y_trainset), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()


def square2row(square, num_obs):
    """Reshape a stack of 28x28 images into (num_obs, 784) flat vectors."""
    return np.resize(square, [num_obs, 28 * 28])


X_train, y_train = square2row(x_trainset, x_trainset.shape[0]), y_trainset
X_test, y_test = square2row(x_test, x_test.shape[0]), y_test
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# Training and prediction time of the three final models on the same machine.
# The tree and forest use the settings chosen in run_analysis.py.
timing = {}
with open(os.path.join(RES_DIR, "metrics.json")) as f:
    tree_params = json.load(f)["decision_tree"]["params"]
tree_params = {k: (None if v == "None" else v) for k, v in tree_params.items()}
for name, model in [("decision_tree", DecisionTreeClassifier(random_state=RNG, **tree_params)),
                    ("random_forest", RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                                             n_jobs=-1, random_state=RNG)),
                    ("random_forest_m112", RandomForestClassifier(n_estimators=200, max_features=112,
                                                                  n_jobs=-1, random_state=RNG))]:
    t = time.time()
    model.fit(X_train, y_train)
    fit_s = time.time() - t
    t = time.time()
    model.predict(X_test)
    timing[name] = {"fit_sec": round(fit_s, 1), "predict_sec": round(time.time() - t, 2)}
    print(name, timing[name], flush=True)

t = time.time()
gb = GradientBoostingClassifier(random_state=RNG, verbose=1)
gb.fit(X_train, y_train)
fit_time = time.time() - t
t = time.time()
gb.predict(X_test)
timing["gradient_boosting"] = {"fit_sec": round(fit_time, 1), "predict_sec": round(time.time() - t, 2)}
timing["n_cpu_cores"] = os.cpu_count()

y_pred = gb.predict(X_test)
staged = [accuracy_score(y_test, p) for p in gb.staged_predict(X_test)]
staged_train = [accuracy_score(y_train, p) for p in gb.staged_predict(X_train)]
rep = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)

# same noisy test sets as in run_analysis.py (same seed)
noise = {}
noise_rng = np.random.default_rng(RNG + 1)
for sd in [0.1, 0.2]:
    X_noisy = np.clip(X_test + noise_rng.normal(0, sd, X_test.shape).astype(np.float32), 0, 1)
    noise[str(sd)] = round(accuracy_score(y_test, gb.predict(X_noisy)), 4)

out = {
    "params": "default: n_estimators=100, learning_rate=0.1, max_depth=3",
    "fit_time_sec": round(fit_time, 1),
    "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
    "test_macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
    "per_class_f1": {c: round(rep[c]["f1-score"], 4) for c in CLASS_NAMES},
    "per_class_recall": {c: round(rep[c]["recall"], 4) for c in CLASS_NAMES},
    "test_acc_per_round": [round(a, 4) for a in staged],
    "train_acc_per_round": [round(a, 4) for a in staged_train],
    "noise_robustness_test_acc": noise,
    "timing": timing,
    "predictions": y_pred.tolist(),
}
with open(os.path.join(RES_DIR, "gradient_boosting.json"), "w") as f:
    json.dump(out, f)
# figure: accuracy per boosting round
FIG_DIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "0.9",
                     "legend.frameon": False, "savefig.bbox": "tight"})
rounds = np.arange(1, len(staged) + 1)
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.plot(rounds, staged_train, color="#4C72B0", label="training")
ax.plot(rounds, staged, color="#C44E52", label="test")
ax.set_xlabel("boosting round")
ax.set_ylabel("accuracy")
ax.set_title("Gradient boosting: accuracy per round")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "gb_rounds.png"), dpi=200)
plt.close()

print("Gradient boosting test accuracy:", out["test_accuracy"], f"({fit_time / 60:.0f} min)")
