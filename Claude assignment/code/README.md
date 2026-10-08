# MLA final project - code

Two scripts:

- `part1_trees.py` - all of Part 1: data analysis (class averages, k-means,
  pixel correlations), decision tree (3-fold CV over max_depth and
  min_samples_leaf, cost-complexity pruning), random forest and bagging
  (3-fold CV over the number of trees and the subset size m), noise check,
  SHAP values of the final forest, pixel selection, gradient boosting with
  default settings and the training times. Writes `../results/part1.json`
  and the figures in `../figures/`.
- `part2_cnn_check.py` - trains the proposed CNN and a fully connected network
  once as a check of the Part 2 design. Writes `../results/part2.json` and
  `../figures/cnn_history.png`.

The data are loaded with `keras.datasets.fashion_mnist.load_data()`, which
downloads Fashion-MNIST the first time.

## How to run

    pip install -r requirements.txt
    python part1_trees.py        # about 2.5 hours on a laptop
    python part2_cnn_check.py    # about 1 hour

All random seeds are fixed (42).
