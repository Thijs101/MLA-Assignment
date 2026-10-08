"""
MLA final project - XGBoost with default settings, on clean and noisy test images.
Uses the same data, seed and noise as part1_trees.py.
Writes ../results/xgboost.json

Usage (from the code/ folder):  python xgboost_check.py
"""
import json
import os
import time

import numpy as np
from tensorflow import keras
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEED = 42
RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RES_DIR, exist_ok=True)

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def square2row(square, num_obs):
    return np.resize(square, [num_obs, 28 * 28])


(x_trainset, y_trainset), (x_testset, y_testset) = keras.datasets.fashion_mnist.load_data()
X_train = square2row(x_trainset, x_trainset.shape[0]).astype(np.float32) / 255.0
y_train = y_trainset
X_test = square2row(x_testset, x_testset.shape[0]).astype(np.float32) / 255.0
y_test = y_testset

noise_rng = np.random.default_rng(SEED + 1)
noisy_tests = {sd: np.clip(X_test + noise_rng.normal(0, sd, X_test.shape).astype(np.float32), 0, 1)
               for sd in [0.1, 0.2]}

xgb = XGBClassifier(n_jobs=-1, random_state=SEED)
start = time.time()
xgb.fit(X_train, y_train)
fit_sec = round(time.time() - start, 1)

y_pred = xgb.predict(X_test)
rep = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
results = {
    "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
    "train_accuracy": round(accuracy_score(y_train, xgb.predict(X_train)), 4),
    "test_macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
    "per_class_f1": {c: round(rep[c]["f1-score"], 4) for c in CLASS_NAMES},
    "noise_robustness_test_acc": {str(sd): round(accuracy_score(y_test, xgb.predict(X)), 4)
                                  for sd, X in noisy_tests.items()},
    "fit_sec": fit_sec,
    "n_cpu_cores": os.cpu_count(),
}
print("XGBoost test accuracy:", results["test_accuracy"])
print("XGBoost with noise:", results["noise_robustness_test_acc"])

with open(os.path.join(RES_DIR, "xgboost.json"), "w") as f:
    json.dump(results, f, indent=2)
