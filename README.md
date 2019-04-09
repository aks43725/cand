# cand

This repository provides three methods to generate candidate models for multi-model inference under the linear model settings.


## Methods to generate candidate models

- The **genetic algorithm (GA)** described in [1].
- The simulated annealing (SA) algorithm introduced in [2].
- The union of models on regularization paths (RP) of the Lasso, SCAD and MCP.

Specific settings of the implementation are described in [1].

The GA has been shown to outperform the SA and the RP in several multi-model inference tools.

### References

- [1] Cheng, C.W. and Cheng, G. (2019+). "Enhancing Multi-model Inference with Natural Selection." (In preparation)
- [2] Nevo, D. and Ritov, Y. (2017), "Identifying a Minimal Class of Models for High-dimensional Data," _Journal of Machine Learning Research_, 18, 1-29.


## Installation

To install this package using command line:

```
git clone https://github.com/aks43725/cand.git
cd cand
python setup.py install
```


## Usage

Import necessary packages and read the riboflavin dataset.
```
import numpy as np
import scipy as sp
import pandas as pd
import cand

riboflavin = pd.read_csv('data/riboflavin.csv', index_col=0).T
y = riboflavin['q_RIBFLV']
X = riboflavin.drop(['q_RIBFLV'], axis=1)
n, d = X.shape

np.random.seed(2549)
```

Implement the GA and prepare model sizes for the SA:

```
mobj_GA = cand.GA(X, y)
mobj_GA.fit()
```

Implement the SA to search for good models of sizes appeared in the last GA generation:
```
count = np.bincount(mobj_GA.generations['model_size'][-1])
SA_sizes = np.nonzero(count)[0]
mobj_SA = cand.SA(X, y)
mobj_SA.fit(SA_sizes)
```

Implement the RP:
```
mobj_RP = cand.union(X, y).fit()
```
