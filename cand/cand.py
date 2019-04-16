import numpy as np
import pandas as pd
import warnings
from sklearn import linear_model
import scipy.stats, pycasso
import matplotlib.pyplot as plt
import glmnet, tqdm
#import multiprocessing as mp

from .utils import GIC, lsIC



class candidate_models:
  def __init__(self, X, y, evaluator='PLIC', varimp_type=None):
    self.X, self.y = np.array(X).copy(), np.array(y).copy()
    self.n, self.d = X.shape
    if not evaluator in ['AIC', 'BIC', 'mBIC', 'eBIC', 'PLIC']:
      self.evaluator = 'PLIC'
      warnings.warn('[evaluator] is set to [PLIC]')
    else:
      self.evaluator = evaluator
    
    self.varimp_type = varimp_type
    if varimp_type is None:
      self.varimp = None
    else:
      self.calculate_varimp()
      self.varimp[self.varimp == 0.0] = min(1e-7, np.min(self.varimp[self.varimp > 0.0]))
  
  def calculate_varimp(self):
    if self.varimp_type == 'marcor':
      self.varimp = np.abs(np.corrcoef(self.y, self.X.T)[0,:][1:])
    elif self.varimp_type == 'holp':
      XX = np.append(np.ones((self.n, 1)), self.X, axis=1)
      if self.n > self.d:
        self.varimp = np.dot(np.linalg.inv(np.dot(XX.T, XX)), XX.T).dot(self.y)[1:]
      else:
        self.varimp = np.dot(XX.T, np.linalg.inv(np.dot(XX, XX.T))).dot(self.y)[1:]
    elif self.varimp_type.upper() == 'NR17':
      # Lasso
      l1_fit = linear_model.LassoCV(n_jobs=-1).fit(self.X, self.y)
      l1_coef = l1_fit.coef_
      l1_model = np.where(np.abs(l1_coef) > 0.0)[0]
      # Elastic net
      en_fit = linear_model.ElasticNetCV(n_jobs=-1).fit(self.X, self.y)
      en_coef = en_fit.coef_
      en_model = np.where(np.abs(en_coef) > 0.0)[0]
      #
      lasso_only = list(set(l1_model) - set(en_model))
      en_only = np.array(list(set(en_model) - set(l1_model)))
      penalties = np.ones(self.X.shape[1])
      delta = np.linspace(0.0, 0.1, 51)
      sims_coef_las_step2 = np.zeros((delta.size, self.X.shape[1]))
      if en_only.size > 0:
        for i in range(len(delta) - 1):
          penalties[en_only] = delta[i]
        
        sims_coef_las_step2[i, :] = glmnet.ElasticNet(alpha=1.0).fit(self.X, self.y, relative_penalties=penalties).coef_
      
      sims_coef_las_step2[-1, :] = l1_coef
      self.varimp = np.zeros(self.X.shape[1])
      if en_only.size > 0:
        for i in en_only:
          coef = sims_coef_las_step2[:, i]
          ij = np.max(np.where(coef != 0.0)[0]) if any(coef != 0.0) else None
          self.varimp[i] = delta[ij] / 2.0 if ij is not None else 0.0
      
      if l1_model.size > 0:
        for i in l1_model:
          coef = sims_coef_las_step2[:, i]
          ij = np.max(np.where(coef == 0.0)[0]) if any(coef == 0.0) else None
          self.varimp[i] = 1.0 - delta[ij] / 2.0 if ij is not None else 1.0
      
    else:
      self.varimp = np.ones(self.d)



class GA(candidate_models):
  '''
  A genetic algorithm for high-quality model search
  '''
  def __init__(self, X, y, evaluator='PLIC', varimp_type='marcor', popSize=0, selection='icmw', mutation_type=None, mutation_rate=None, ggap=15, maxGen=100):
    candidate_models.__init__(self, X, y, evaluator, varimp_type)
    if popSize <= 0:
      popSize = int(4 * np.ceil(1 + np.log(-X.shape[1] / np.log(0.9999)) / np.log(2)))
    self.popSize, self.selection = popSize, selection
    if mutation_type is None:
      self.mutation_type = 'uniform'
    else:
      self.mutation_type = mutation_type
    
    if mutation_rate is None:
      self.mutation_rate = 1.0 / X.shape[1]
    else:
      self.mutation_rate = mutation_rate
    
    self.worst_fitness = np.nan
    
    # Results
    self.models = None
    self.fitness = None
    self.generations = {'fitness': [], 'model_size': [], 'model_history': []}
    self.ggap = ggap
    self.maxGen = maxGen
  
  def get_fitness(self, models, n_jobs=-1):
    # Negative GIC as fitness
    fitness = -GIC(self.X, self.y, models, self.evaluator, n_jobs)
    
    # Update 'worst_fitness'
    self.worst_fitness = np.nanmin(np.append(self.worst_fitness, fitness))
    return np.where(pd.isnull(fitness), self.worst_fitness, fitness)
  
  def uniform_xover(self, models, prob_select=None):
    if prob_select.sum() is not 1.0:
      prob_select = prob_select / prob_select.sum()
    
    m = models[np.random.choice(range(models.shape[0]), size=2, replace=True, p=prob_select), :]
    if any(m[0,:] != m[1,:]):
      idx = np.where(m[0,:] != m[1,:])[0]
      idxx = np.random.binomial(1, 0.5, size=idx.size)
      if idxx.sum() > 0:
        m[0, idx[idxx == 1]], m[1, idx[idxx == 1]] = m[1, idx[idxx == 1]], m[0, idx[idxx == 1]]
    
    # Mutation
    for k in range(m.shape[0]):
      prob_mutate = np.repeat(self.mutation_rate, self.d)
      if (self.mutation_type == 'varimp') and (np.min(self.varimp) != np.max(self.varimp)):
        idx0 = np.where(m[k, :] == 0)[0]
        prob_mutate[idx0] *= idx0.size * self.varimp[idx0] / np.sum(self.varimp[idx0])
        idx1 = np.where(m[k, :] == 1)[0]
        prob_mutate[idx1] *= idx1.size / self.varimp[idx1] / np.sum(1.0 / self.varimp[idx1])

      idx_mutate = np.where(np.random.random(m[k, :].size) < self.mutation_rate)[0]
      if idx_mutate.size > 0:
        m[k, idx_mutate] = 1 - m[k, idx_mutate]
    
    return m
  
  def fit(self, init_models='RP', model_history=False, seed=None, verbose=False, n_jobs=-1):
    if seed is not None:
      np.random.seed(seed)
    
    # Generate initial population, fitness evaluation, and sort the models by fitness
    if isinstance(init_models, np.ndarray):
      self.models = 1 * (init_models != 0)
    elif init_models == 'RP':
      self.models = RP(self.X, self.y, None, None, 'RP').fit()
    else:
      '''
      Random initial model generation
      Using HyperGeometric distribution to determine the model sizes and 
      then randomly assign active positions based on variable importance
      '''
      init_model_sizes = scipy.stats.hypergeom(6 * min(self.n, self.d), 2 * min(self.n, self.d), min(self.n, self.d)).rvs(size=self.popSize)
      self.models = np.zeros((self.popSize, self.d)).astype('int')
      for k in range(self.popSize):
        self.models[k, np.random.choice(self.d, size=init_model_sizes[k], replace=False, p=self.varimp/self.varimp.sum())] = 1
    
    self.fitness = self.get_fitness(self.models, n_jobs)
    idx_sort = np.argsort(self.fitness)[::-1]
    self.models, self.fitness = self.models[idx_sort,], self.fitness[idx_sort]
    
    # Generation summary
    self.generations['fitness'].append(self.fitness)
    self.generations['model_size'].append(self.models.sum(axis=1))
    if model_history:
      self.generations['model_history'].append(self.models)
    
    # Updating the population
    converge = False
    it = 0
    while not converge and (it <= self.maxGen):
      # Check whether all current models are infeasible
      if np.unique(self.fitness).size == 1:
        raise RuntimeError('All models in generation {} are infeasible'.format(it))
      it += 1
      if verbose:
        print('\t{} generations'.format(it), end='\r')
      
      old_models = self.models.copy()
      # Elitism selection: keep the best model
      self.models = old_models[0, :]
      if len(set(self.fitness)) == 1:
        # If fitness values are all the same, it doesn't matter using which model weighting
        prob_select = np.ones(self.popSize) / self.popSize
      elif self.selection == 'icmw':
        # Proportionate selection via GIC-based model weighting
        prob_select = lsIC(self.X, self.y, old_models, self.evaluator, ic=-self.fitness, n_jobs=n_jobs)
      else:
        # Rank selection
        fitness_rank = scipy.stats.rankdata(self.fitness, method='ordinal')
        prob_select = 2.0 * fitness_rank.astype('float') / fitness_rank.size / (fitness_rank.size + 1.0)
      # Uniform crossover and mutation
      children = np.vstack([self.uniform_xover(old_models, prob_select) for _ in range(self.popSize // 2)])
      #children = np.vstack([self.uniform_xover(old_models, prob_select) for _ in range(self.popSize - 1)])
      self.models = np.vstack((self.models, children))
      
      # Fitness evaluation and sort the models by fitness
      self.fitness = self.get_fitness(self.models, n_jobs)
      idx_sort = np.argsort(self.fitness)[::-1][:self.popSize]
      self.models, self.fitness = self.models[idx_sort, :], self.fitness[idx_sort]
      
      # Generation summary
      self.generations['fitness'].append(self.fitness)
      self.generations['model_size'].append(self.models.sum(axis=1))
      if model_history:
        self.generations['model_history'].append(self.models)
      
      # Check convergence
      if it > self.ggap:
        converge = (scipy.stats.ttest_ind(self.generations['fitness'][it], self.generations['fitness'][it - self.ggap], equal_var=False)[1] >= 0.05)
    
    # Remove the duplicated models in the last generation
    self.models, idx = np.unique(self.models, return_index=True, axis=0)
    self.fitness = self.fitness[idx]
    idx = np.argsort(self.fitness)[::-1]
    self.models = self.models[idx, :]
    self.fitness = self.fitness[idx]
    return 
  
  def plot_fitness(self, file=None, true_model=None, n_jobs=-1):
    plt.plot(list(map(np.mean, self.generations['fitness'])), label='Average Fitness', color='forestgreen')
    plt.plot(list(map(np.max, self.generations['fitness'])), label='Best Fitness', color='steelblue')
    if true_model is not None:
      plt.axhline(self.get_fitness(true_model.reshape(1, true_model.size), n_jobs), label='True Model', c='tomato', linestyle='--')
    
    plt.legend(loc = 4)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    if file is not None:
      plt.savefig(file + '.pdf', bbox_inches='tight')
    else:
      plt.show()
    
    plt.cla()
    plt.clf()
    plt.close('all')



class SA(candidate_models):
  '''
  Simulated annealing algorithm of
    Daniel Nevo, Ya'acov Ritov (2017)
    Identifying a Minimal Class of Models for High-dimensional Data
    Journal of Machine Learning Research, 18(24):1-29
  '''
  def __init__(self, X, y, evaluator='PLIC', varimp_type='NR17', popSize=100, max_iter=100):
    candidate_models.__init__(self, X, y, evaluator, varimp_type)
    self.popSize, self.max_iter = popSize, max_iter
  
  def generate_init_model(self, model_size):
    '''
    Auxiliary function to choose initial model
    '''
    if model_size > self.varimp.size:
      warnings.warn('[model_size] should be at most {:d}'.format(self.X.shape[1]))
      return 
    
    model = np.zeros(self.varimp.size).astype('int')
    taken = np.where(self.varimp > np.sort(self.varimp)[-model_size])[0]
    if taken.size > 0:
      model[taken] = 1
    
    nfree = model_size - taken.size
    idx = np.where(self.varimp == np.sort(self.varimp)[-model_size])[0]
    if idx.size > 0:
      model[np.random.choice(idx, nfree, replace=False)] = 1
    
    return model
  
  def metropolis_iter(self, old_model, BoltzmanT, n_jobs=-1):
    '''
    Iteration for fixed temperature (BoltzmanT), a model is suggested and then a decision is made
    '''
    new_model = old_model.copy()
    # Choose a variable to be kicked out
    prob_old_out = np.zeros(self.X.shape[1])
    prob_old_out[old_model == 1] = 1.0 / self.varimp[old_model == 1]
    prob_old_out = np.where(pd.isnull(prob_old_out), 0.0, prob_old_out)
    if np.all(prob_old_out == 0.0):
      prob_old_out = np.ones(prob_old_out.size) / prob_old_out.size
    else:
      prob_old_out = prob_old_out / prob_old_out.sum()
    var_out = np.random.choice(range(self.X.shape[1]), 1, p=prob_old_out)
    # Choose a variable to be added in
    prob_old_in = np.zeros(self.X.shape[1])
    prob_old_in[old_model == 0] = self.varimp[old_model == 0]
    prob_old_in = np.where(pd.isnull(prob_old_in), 0.0, prob_old_in)
    if np.all(prob_old_in == 0.0):
      prob_old_in = 1.0 / prob_old_out.size
    else:
      prob_old_in = prob_old_in / prob_old_in.sum()
    prob_old_in = prob_old_in / prob_old_in.sum()
    var_in = np.random.choice(range(self.X.shape[1]), 1, p=prob_old_in)
    new_model[var_out], new_model[var_in] = 0, 1
    #
    # Probabilities from the new model back to the old model
    prob_new_in = np.zeros(self.X.shape[1])
    prob_new_in[new_model == 0] = self.varimp[new_model == 0]
    prob_new_in = prob_new_in / prob_new_in.sum()
    prob_new_out = np.zeros(self.X.shape[1])
    prob_new_out[new_model == 1] = 1.0 / self.varimp[new_model == 1]
    prob_new_out = prob_new_out / prob_new_out.sum()
    #
    # GIC values of the old and new models
    ic = GIC(self.X, self.y, np.array([old_model, new_model]), self.evaluator, n_jobs)
    #
    # Whether to accept the new model
    decision_prob = np.exp((ic[1] - ic[0]) / BoltzmanT) * prob_new_out[var_in] * prob_new_in[var_out] / prob_old_out[var_out] / prob_old_in[var_in]
    decision_prob = min(1.0, decision_prob)
    decision = 1 if np.random.uniform() < decision_prob else 0
    return {'decision': decision, 'model': np.array([old_model, new_model]), 'ic': ic}
  
  def run_constant_temperature(self, init_model, BoltzmanT, models, ic, n_jobs=-1):
    '''
    init_model: Initial model
    current_ic: GIC of the current best models
    current_top_models: Current best models
    BoltzmanT: Temperature sequence
    max_init: Number of iterations per temperature
    '''
    if models.shape[0] == 0:
      models = np.array(init_model)
      ic = GIC(self.X, self.y, models, self.evaluator, n_jobs)
    
    for it in range(self.max_iter):
      met_res = self.metropolis_iter(init_model, BoltzmanT, n_jobs)
      if met_res['decision']:
        init_model = met_res['model'][1, :]
        if ic.size < self.popSize:
          models = np.vstack((met_res['model'][1, :], models))
          ic = np.append(met_res['ic'][1], ic)
        else:
          models[-1, :] = met_res['model'][1, :]
          ic[-1] = met_res['ic'][1]
        
        idx = np.argsort(ic)
        models, ic = models[idx, :], ic[idx]

    return models, ic, init_model

  def fit(self, model_sizes, T_seq=None, verbose=False, n_jobs=-1):
    self.models = np.empty((0, self.X.shape[1]), 'int')
    self.ic = np.array([])
    if T_seq is None:
      T_seq = 10.0 * 0.7**np.linspace(1, 20, 20)
    for k in model_sizes:
      if verbose:
        print('SA with model size {:d}'.format(k))
      
      if k > self.varimp.size:
        warnings('Cannot process model size {:d}'.format(k))
        continue
      
      #init_model = self.generate_init_model(k)
      init_model = np.zeros(self.X.shape[1]).astype('int')
      init_model[:k] = 1
      #
      # Simulated annealing algorithm
      old_model = init_model.copy()
      models_k = np.empty((0, self.X.shape[1]), 'int')
      ic_k = np.array([])
      for t in tqdm.tqdm(T_seq) if verbose else T_seq:
        models_k, ic_k, old_model = self.run_constant_temperature(old_model, t, models_k, ic_k, n_jobs)
      
      self.models = np.vstack((self.models, models_k))
      self.ic = np.append(self.ic, ic_k)
    
    # Remove duplicate models and sort the models by the their GIC values
    idx = np.argsort(self.ic)
    if self.ic.size > self.popSize:
      self.models, self.ic = self.models[idx[:self.popSize], :], self.ic[idx[:self.popSize]]
    else:
      self.models, self.ic = self.models[idx, :], self.ic[idx]
    
    self.models, idx = np.unique(self.models, return_index=True, axis=0)
    self.ic = self.ic[idx]
    idx = np.argsort(self.ic)
    if self.ic.size > self.popSize:
      self.models, self.ic = self.models[idx[:self.popSize], :], self.ic[idx[:self.popSize]]
    else:
      self.models, self.ic = self.models[idx, :], self.ic[idx]
    
    return



class RP(candidate_models):
  '''
  The RP (regularization paths) method for candidate model preparation
  '''
  def __init__(self, X, y, evaluator='None', varimp_type='None', method='RP'):
    candidate_models.__init__(self, X, y, 'PLIC', varimp_type)
    if method not in ['RP', 'marcor', 'holp']:
      self.method = 'RP'
      warnings.warn('[method] is set to [RP]')
    else:
      self.method = method
  
  def fit(self, seed=None):
    if seed is not None:
      np.random.seed(seed)
    
    n, d = self.X.shape
    if self.method == 'RP':
      models = np.array([]).reshape(0, d)
      for penalty in ['l1', 'scad', 'mcp']:
        ncpfit = pycasso.Solver(self.X, self.y, penalty=penalty, lambdas=[200, 0.001], useintercept=True)
        ncpfit.train()
        models = np.vstack([models, np.unique(1 * (ncpfit.coef()['beta'] != 0.0), axis=0)])
      
      models = models[np.where(models.sum(axis=1) < n - 1)[0], :].astype('int')
      models = np.unique(models, axis=0)
    else:
      if method == 'marcor':
        varimp = np.abs(np.corrcoef(self.y, self.X.T)[0, :][1:])
      elif method == 'holp':
        XX = np.append(np.ones((n, 1)), self.X, axis=1)
        varimp = np.dot(XX.T, np.linalg.inv(np.dot(XX, XX.T))).dot(y)[1:]
      else:
        varimp = np.ones(d)
      
      models = np.zeros((n - 2) * d).astype('int').reshape(n - 2, d)
      order = np.argsort(varimp)[::-1]
      for k in range(models.shape[0]):
        models[k, order[:(k + 1)]] = 1
    
    return models

