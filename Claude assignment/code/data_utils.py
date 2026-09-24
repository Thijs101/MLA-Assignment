"""
Loader for the Fashion-MNIST dataset (Xiao, Rasul & Vollgraf, 2017).

Reads the original IDX ubyte format directly, following the format
description at https://github.com/zalandoresearch/fashion-mnist.
"""
import gzip
import os
import numpy as np

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _read_idx_images(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    assert magic == 2051, f"unexpected magic number {magic} in {path}"
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    arr = np.frombuffer(data, dtype=np.uint8, offset=16)
    return arr.reshape(n, rows * cols)


def _read_idx_labels(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    assert magic == 2049, f"unexpected magic number {magic} in {path}"
    n = int.from_bytes(data[4:8], "big")
    arr = np.frombuffer(data, dtype=np.uint8, offset=8)
    assert arr.shape[0] == n
    return arr


def load_fashion_mnist(data_dir):
    X_train = _read_idx_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    y_train = _read_idx_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    X_test = _read_idx_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    y_test = _read_idx_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
    return X_train, y_train, X_test, y_test
