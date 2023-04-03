# Kernel methods for protein classification

**Author:** Jérémie Dentan

## Overview

This repository contains the implementation of the work of the author for a data challenge. The goal of this challenge is to apply kernel methods on protein graph for classification.

For more details about our work, please refer to:

- The competition: [https://www.kaggle.com/competitions/data-challenge-kernel-methods-2022-2023](https://www.kaggle.com/competitions/data-challenge-kernel-methods-2022-2023)
- Our report: [doc/Kernel_methods_for_protein_classification.pdf](doc/Kernel_methods_for_protein_classification.pdf)

## Re-running our experiments

Our code is expected to run in **Python 3.9** with the dependencies installed and the PYTHONPATH set to the root of the repository:

```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

The, you may run our experiments, which will download some data and compute some cache. You will find some logs in `logs` to understand the experiments, and the final prediction will be stored at the root, under `test_pred.csv`:

```bash
python start.py
```

## License and disclaimer

You may use this software under the Apache 2.0 License. See LICENSE.
