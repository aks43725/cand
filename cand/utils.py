'''
Auxiliary functions for simulation
'''
import numpy as np
import scipy as sp
import os, warnings
import scipy.linalg, scipy.sparse, scipy.misc
import quadprog
from joblib import Parallel, delayed

'''
Generalized information criteria (GIC)
'''
def GIC(X, y, models, evaluator, n_jobs=1):
  '''
  Generalized information criteria
  '''
  if evaluator not in ['AIC', 'BIC', 'mBIC', 'eBIC', 'PLIC']:
    warnings.warn('[evaluator] is not valid')
    return 
  
  n_jobs = os.cpu_count() if n_jobs <= 0 or n_jobs > os.cpu_count() else int(n_jobs)
  X, y = np.array(X), np.array(y)
  n, d = X.shape
  if models.ndim == 1:
    models = models.reshape((1, d))
  
  d_hat = models.sum(axis=1)
  K = models.shape[0]
  
  def compute_rss(X, y, model):
    n, d = X.shape
    model = np.array(model)
    #if model.sum() - 1 < n:
    #  return np.linalg.lstsq(np.hstack([np.ones(n).reshape((n, 1)), X[:, model == 1]]), y, rcond=0)[1][0]
    #else:
    #  return np.nan
    try:
      return np.linalg.lstsq(np.hstack([np.ones(n).reshape((n, 1)), X[:, model == 1]]), y, rcond=0)[1][0]
    except:
      return np.nan
  
  if n_jobs > 1:
    RSS = np.array(Parallel(n_jobs=n_jobs, backend='threading')(delayed(compute_rss)(X, y, m) for m in models.tolist()))
  else:
    RSS = np.array([np.linalg.lstsq(np.hstack([np.ones(n).reshape((n, 1)), X[:, (models[k, :] == 1)]]), y, rcond=0)[1] for k in range(K)]).reshape(-1)
  
  # Negative GIC as fitness
  if evaluator == 'AIC':
    penalty = 2.0 * d_hat
  elif evaluator == 'BIC':
    penalty = d_hat * np.log(n)
  elif evaluator == 'mBIC':
    penalty = np.array([np.log(np.log(d_hat[k])) * d_hat[k] * np.log(n) if d_hat[k] > 2 else 0.0 for k in range(K)])
    #penalty = np.where(d_hat > 2, np.log(np.log(d_hat)) * d_hat * np.log(n), 0.0)
  elif evaluator == 'eBIC':
    penalty = d_hat * (np.log(n) + 2 * np.log(d))
    #penalty = d_hat * np.log(n) + 2.0 * np.log(sp.misc.comb(d, d_hat))
  else:
    penalty = 3.5 * d_hat * np.log(d)
  
  #val = np.where(d_hat < n - 1, n * np.log(RSS / (n - d_hat - 1)), float('nan')) + penalty
  val = np.array([n * np.log(RSS[k] / (n - d_hat[k] - 1)) if d_hat[k] < n - 1 else float('nan') for k in range(K)]) + penalty
  #return np.where(np.isinf(val), float('nan'), val)
  return val

'''
Calculation of model weights
'''
def lsIC(X, y, models, weighting, ic=None, n_jobs=1):
  if not weighting in ['AIC', 'BIC', 'mBIC', 'eBIC', 'PLIC', 'AL14']:
    warnings.warn('[weighting] is not valid')
    return 
  
  if weighting in ['AIC', 'BIC', 'mBIC', 'eBIC', 'PLIC']:
    '''
    Information criteria based models weights
    '''
    if ic is None:
      ic = GIC(X, y, models, weighting, n_jobs)
    ic = ic.astype('float')
    prob_select = np.exp((np.min(ic) - ic) / 2.0)
    return prob_select / prob_select.sum()
  else:
    '''
    Optimal high-dimensional model averaging of Ando and Li (2014, JASA)
    '''
    n, K = y.size, models.shape[0]
    a, B = np.zeros(K), np.zeros((K, K))
    Hy = []
    for k in range(K):
      u = scipy.sparse.linalg.svds(np.hstack([np.ones(n).reshape((n, 1)), X[:, (models[k, :] == 1)]]), models[k, :].sum())[0]
      H = np.dot(u, u.T)
      D = np.diag(1.0 / (1.0 - np.diag(H)))
      Hy.append(np.dot(np.dot(D, H - np.identity(n)) + np.identity(n), y))
    
    a = np.array([np.sum(y * Hy[k]) for k in range(K)])
    B = np.dot(np.array(Hy), np.array(Hy).T)
    wts = quadprog.solve_qp(B, a, np.identity(K), np.zeros(K))[0]
    return wts

'''
Simulation data generation
'''
def gendata(n, d, s, sim_case, param={'rho': 0.0, 'sig2': 1.0}, seed=None):
  if sim_case not in (np.arange(6) + 1):
    warnings.warn("Input 'sim_case' is incorrect")
    return
  
  if s < 6:
    s = 6
    warnings.warn('s is set to 6')
  
  if seed is not None:
    np.random.seed(seed)
  
  # True beta values
  a = 3.0 * np.log(n) / np.sqrt(n)
  if sim_case in [1, 2]:
    beta0 = np.append(np.repeat(4.0, s - 2), [-6.0 * np.sqrt(2.0), 4.0 / 3.0])
  elif sim_case == 3:
    beta0 = np.repeat(3.0, s)
  elif sim_case == 4: # Weak signals
    beta0 = np.repeat(a, s)
  elif sim_case == 5:
    beta0 = np.random.uniform(0.5, 1.5, s)
    beta0[::-1].sort()
  else: # Weak signals
    beta0 = np.repeat(a, s)
  
  beta0 = np.append(beta0, np.zeros(d - s))
  # Design matrix
  # Case1 to Case4 are Toeplitz, where Case2-4 have special covariates
  # Case5 and Case6 are exchangeable
  if sim_case in (np.arange(4) + 1):
    Sig = sp.linalg.toeplitz(param['rho']**(np.arange(d)))
  else:
    Sig = np.tile(param['rho'], (d, d))
    np.fill_diagonal(Sig, 1.0)
  
  X = np.random.multivariate_normal(np.zeros(d), Sig, size=n)
  # Special covariates
  if sim_case == 2:
    X[:, s] = 0.5 * X[:, 0] + 2.0 * X[:, s - 2] + np.random.normal(scale=0.1, size=n)
  
  if sim_case in [3, 4]:
    X[:, s] = 2.0 * (X[:, 0] + X[:, 1]) / 3.0 / np.sqrt(1.0 + param['rho']) + np.random.normal(size=n) / 3.0
    X[:, s + 1] = 2.0 * (X[:, 2] + X[:, 3]) / 3.0 / np.sqrt(1.0 + param['rho']) + np.random.normal(size=n) / 3.0
  
  # Response
  y = 1.0 + np.dot(X[:,:s], beta0[:s]) + np.random.normal(scale=np.sqrt(param['sig2']), size=n)
  return {'y': y, 'X': X, 'beta0': beta0, 'sig2': param['sig2']}


