"""
MLA final project - Part 2: one training run of the proposed CNN.

Part 2 is a design; this script only checks that the design works. It trains the
CNN from Table 2 of the report once (no tuning) and a fully connected network
(784-150-60-10) for comparison, on a stratified 10% validation split of the
training set, and evaluates both on the test set.
Writes ../results/part2.json and ../figures/cnn_history.png (about 1 hour on a laptop).

Usage (from the code/ folder):  python part2_cnn_check.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Dense, Conv2D, MaxPooling2D, Flatten,
                                     Dropout, RandomFlip, RandomTranslation)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEED = 42
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
keras.utils.set_random_seed(SEED)

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def square2row(square, num_obs):
    return np.resize(square, [num_obs, 28 * 28])


(x_trainset, y_trainset), (x_testset, y_testset) = keras.datasets.fashion_mnist.load_data()
X_train = square2row(x_trainset, x_trainset.shape[0]).astype("float32") / 255.0
X_test = square2row(x_testset, x_testset.shape[0]).astype("float32") / 255.0
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_trainset, test_size=0.10,
                                            stratify=y_trainset, random_state=SEED)

mlp = Sequential([
    Input(shape=(784,)),
    Dense(150, activation="relu"),
    Dense(60, activation="relu"),
    Dense(10, activation="softmax"),
])

cnn = Sequential([
    Input(shape=(28, 28, 1)),
    RandomFlip("horizontal"),                               # only active while training
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

results, histories = {}, {}
for name, model, shape in [("mlp", mlp, (-1, 784)), ("cnn", cnn, (-1, 28, 28, 1))]:
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_tr.reshape(shape), to_categorical(y_tr, 10), epochs=30, batch_size=32,
                        validation_data=(X_val.reshape(shape), to_categorical(y_val, 10)),
                        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
                        verbose=2)
    y_pred = model.predict(X_test.reshape(shape), verbose=0).argmax(axis=1)
    rep = classification_report(y_testset, y_pred, target_names=CLASS_NAMES, output_dict=True)
    histories[name] = history.history
    results[name] = {"n_params": int(model.count_params()),
                     "epochs_run": len(history.history["loss"]),
                     "test_accuracy": round(accuracy_score(y_testset, y_pred), 4),
                     "test_macro_f1": round(f1_score(y_testset, y_pred, average="macro"), 4),
                     "per_class_f1": {c: round(rep[c]["f1-score"], 4) for c in CLASS_NAMES}}
    print(name, "test accuracy:", results[name]["test_accuracy"])

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "0.9",
                     "legend.frameon": False, "savefig.bbox": "tight"})
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
for name, col in [("mlp", "#4C72B0"), ("cnn", "#C44E52")]:
    h = histories[name]
    epochs = np.arange(1, len(h["loss"]) + 1)
    axes[0].plot(epochs, h["loss"], color=col, ls="--", label=f"{name.upper()} training")
    axes[0].plot(epochs, h["val_loss"], color=col, label=f"{name.upper()} validation")
    axes[1].plot(epochs, h["accuracy"], color=col, ls="--", label=f"{name.upper()} training")
    axes[1].plot(epochs, h["val_accuracy"], color=col, label=f"{name.upper()} validation")
axes[0].set_title("Loss")
axes[1].set_title("Accuracy")
for ax in axes:
    ax.set_xlabel("epoch")
axes[1].legend(fontsize=7.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cnn_history.png"), dpi=200)
plt.close()

with open(os.path.join(RES_DIR, "part2.json"), "w") as f:
    json.dump(results, f, indent=2)
