# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import pandas as pd
import cand

riboflavin = pd.read_csv('data/riboflavin.csv', index_col=0).T
y = riboflavin['q_RIBFLV']
X = riboflavin.drop(['q_RIBFLV'], axis=1)
n, d = X.shape

np.random.seed(2549)

if __name__ == '__main__':
  
  # Implementing the GA
  mobj_GA = cand.GA(X, y)
  mobj_GA.fit()
  
  # Implementing the SA to search for good models of sizes appeared in the last GA generation
  count = np.bincount(mobj_GA.generations['model_size'][-1])
  SA_sizes = np.nonzero(count)[0]
  mobj_SA = cand.SA(X, y)
  mobj_SA.fit(SA_sizes)
  
  # Implementing the RP:
  mobj_RP = cand.union(X, y).fit()
