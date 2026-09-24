# MLA assignment - Part 1 code

Reproduces every number and figure in the report (decision tree and random
forest on Fashion-MNIST).

## Files

- `data_utils.py` - reads the four Fashion-MNIST `idx-ubyte.gz` files into
  numpy arrays.
- `run_analysis.py` - full pipeline: loads the data, tunes and evaluates the
  decision tree (Part 1a), tunes and evaluates the random forest (Part 1b),
  fits a gradient boosting model as a reference point for the ensemble-choice
  discussion, and writes all figures and metrics to disk.
- `requirements.txt` - Python package versions used.

## How to run

1. Download the four Fashion-MNIST files from
   https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
   (`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`,
   `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`) into a `data/`
   folder next to this `code/` folder.
2. `pip install -r requirements.txt`
3. `python run_analysis.py`

This writes:

- `../results/metrics.json` - every number quoted in the report.
- `../figures/*.png` - every figure used in the report.

Total runtime on a standard laptop CPU is roughly 10 minutes, dominated by
fitting the 200-tree random forest and the gradient boosting reference model.
