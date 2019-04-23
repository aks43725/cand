# cand

This repository provides three methods to generate candidate models for multi-model inference under the linear model settings.


## Methods to generate candidate models

- The **genetic algorithm (GA)** described in [1].
- The simulated annealing (SA) algorithm introduced in [2].
- The union of distinct models on the regularization paths (RPs) of the Lasso, SCAD and MCP.

Specific settings of the implementation are described in [1].

The GA has been shown to outperform the SA and the RP in several multi-model inference tools.

### References

- [1] Cheng, C.W. and Cheng, G. (2019+). "Enhancing Multi-model Inference with Natural Selection." (Under review)
- [2] Nevo, D. and Ritov, Y. (2017), "Identifying a Minimal Class of Models for High-dimensional Data," _Journal of Machine Learning Research_, 18, 1-29.


## Installation

To install this package using command line:

```shell
git clone https://github.com/aks43725/cand.git
cd cand
python setup.py install
```


## Usage

Import necessary packages and read the riboflavin dataset.
```python
import numpy as np
import scipy as sp
import pandas as pd
import cand

ribflv = pd.read_csv('data/riboflavin.csv', index_col=0).T
y = ribflv['q_RIBFLV']
X = ribflv.drop(['q_RIBFLV'], axis=1)
n, d = X.shape

np.random.seed(570)
```

Implement the GA and prepare model sizes for the SA:

```python
models_GA = cand.GA(X, y)
models_GA.fit()
print('Model sizes:\n{}'.format(models_GA.models.sum(axis=1)))
print('Fitness values:\n{}'.format(models_GA.fitness))
```

Implement the SA to search for good models of sizes appeared in the last GA generation:
```python
count = np.bincount(models_GA.generations['model_size'][-1])
SA_sizes = np.nonzero(count)[0]
models_SA = cand.SA(X, y)
models_SA.fit(SA_sizes)
print('Model sizes:\n{}'.format(models_SA.models.sum(axis=1)))
print('GIC values:\n{}'.format(models_SA.ic))
```

Implement the RP:
```python
models_RP = cand.RP(X, y)
models_RP.fit()
print('Model sizes:\n{}'.format(models_RP.models.sum(axis=1)))
print('GIC values:\n{}'.format(cand.GIC(X, y, models_RP.models, 'PLIC', n_jobs=-1)))
```
