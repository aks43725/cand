# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import pandas as pd
import cand

ribflv = pd.read_csv('data/riboflavin.csv', index_col=0).T
y = ribflv['q_RIBFLV']
X = ribflv.drop(['q_RIBFLV'], axis=1)
n, d = X.shape


#if __name__ == '__main__':

# The GA
for seed in range(1, 100):
  np.random.seed(111)
  models_GA = cand.GA(X, y)
  models_GA.fit()
  print(models_GA.fitness)
  print(models_GA.models.sum(axis=1))
  if models_GA.models.sum(axis=1)[0] == 1:
    break

# The SA, with model sizes appeared in the last GA generation
count = np.bincount(mobj_GA.generations['model_size'][-1])
SA_sizes = np.nonzero(count)[0]
models_SA = cand.SA(X, y)
models_SA.fit(SA_sizes, verbose=True)
print(models_SA.fitness)
print(models_SA.models.sum(axis=1))

# The RP:
models_RP = cand.union(X, y).fit()
