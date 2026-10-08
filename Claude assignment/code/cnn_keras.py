"""
MLA final project, Part 2 (optional check of the design).

Part 2 of the assignment is design-only. This script trains the proposed CNN
(Table 2 of the report) once in Keras, together
with a fully connected network (784-150-60-10) as a baseline,
to check that the design behaves as we expect. It is not tuned.

Run from the code/ folder:   python cnn_keras.py
Writes: ../results/cnn_keras.json and ../figures/cnn_history.png
"""
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Dense, Conv2D, MaxPooling2D, Flatten,
                                     Dropout, RandomFlip, RandomTranslation)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

RNG = 42
HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
RES_DIR = os.path.join(HERE, "..", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
keras.utils.set_random_seed(RNG)

# Load Fashion-MNIST from Keras (downloads on first run if not cached)
(x_trainset, y_trainset), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()


def square2row(square, num_obs):
    """Reshape a stack of 28x28 images into (num_obs, 784) flat vectors."""
    return np.resize(square, [num_obs, 28 * 28])


X_train_full, y_train_full = square2row(x_trainset, x_trainset.shape[0]), y_trainset
X_test, y_test = square2row(x_test, x_test.shape[0]), y_test
X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# same 90/10 split as in run_analysis.py
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.10, stratify=y_train_full, random_state=RNG)
y_tr_cat, y_val_cat = to_categorical(y_tr, 10), to_categorical(y_val, 10)

# fully connected network (784-150-60-10) on the flattened pixels
mlp = Sequential([
    Input(shape=(784,)),
    Dense(150, activation="relu"),
    Dense(60, activation="relu"),
    Dense(10, activation="softmax"),
])

# proposed CNN; the two augmentation layers are only active during training
cnn = Sequential([
    Input(shape=(28, 28, 1)),
    RandomFlip("horizontal"),
    RandomTranslation(2 / 28, 2 / 28, fill_mode="constant"),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(150, activation="relu", kernel_regularizer=l2(1e-4)),
    Dropout(0.5),
    Dense(10, activation="softmax"),
])

results, histories, preds = {}, {}, {}
for name, model, reshape in [("mlp", mlp, (-1, 784)), ("cnn", cnn, (-1, 28, 28, 1))]:
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    t = time.time()
    hist = model.fit(X_tr.reshape(reshape), y_tr_cat, epochs=30, batch_size=32,
                     validation_data=(X_val.reshape(reshape), y_val_cat),
                     callbacks=[stop], verbose=2)
    fit_time = time.time() - t
    y_pred = model.predict(X_test.reshape(reshape), verbose=0).argmax(axis=1)
    preds[name] = y_pred
    histories[name] = hist.history
    rep = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    results[name] = {
        "n_params": int(model.count_params()),
        "epochs_run": len(hist.history["loss"]),
        "best_epoch": int(np.argmin(hist.history["val_loss"]) + 1),
        "fit_time_min": round(fit_time / 60, 1),
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
        "per_class_f1": {c: round(rep[c]["f1-score"], 4) for c in CLASS_NAMES},
        "per_class_recall": {c: round(rep[c]["recall"], 4) for c in CLASS_NAMES},
        "history": {k: [round(float(v), 4) for v in vals] for k, vals in hist.history.items()},
    }
    print(name, results[name]["test_accuracy"], flush=True)

cm = confusion_matrix(y_test, preds["cnn"])
results["cnn"]["confusion"] = cm.tolist()
off = cm - np.diag(np.diag(cm))
up = [0, 2, 4, 6]
results["cnn"]["errors_within_upper_body"] = round(float(off[np.ix_(up, up)].sum() / off.sum()), 3)

# training and validation curves
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "0.9",
                     "legend.frameon": False, "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for name, col in [("mlp", "#4C72B0"), ("cnn", "#C44E52")]:
    h = histories[name]
    ep = np.arange(1, len(h["loss"]) + 1)
    axes[0].plot(ep, h["loss"], color=col, ls="--", label=f"{name.upper()} training")
    axes[0].plot(ep, h["val_loss"], color=col, label=f"{name.upper()} validation")
    axes[1].plot(ep, h["accuracy"], color=col, ls="--", label=f"{name.upper()} training")
    axes[1].plot(ep, h["val_accuracy"], color=col, label=f"{name.upper()} validation")
axes[0].set_title("Loss")
axes[1].set_title("Accuracy")
for ax in axes:
    ax.set_xlabel("epoch")
axes[1].legend(fontsize=7.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cnn_history.png"), dpi=200)
plt.close()

with open(os.path.join(RES_DIR, "cnn_keras.json"), "w") as f:
    json.dump(results, f, indent=2)
print("done")
