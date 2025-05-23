from .experiment import BenchmarkExperiment
from ..optimizers import OptimizerFactory
from ..search_spaces import RealSearchSpace
from ..benchmarks import BenchQueryResult
import numpy as np

class QuadraticExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, n_dim=3, seed=0) -> None:
        benchmark = None
        self._seed = seed
        self._optimizer_name = optimizer_name
        self.search_space = RealSearchSpace(n_dims=n_dim)
        self.c = [1.4, 1.4, 1.4]
        self.optimizer_config = self.get_optimizer_config()
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)
        super().__init__(benchmark, optimizer)

    def run(self):
        config, performance = self.optimizer.optimize()
        print((config, performance))

    def evaluate_config(self, cfg):
        config_vec = list(cfg.values())
        return quadartic(config_vec, self.c)

    def get_optimizer_config(self):
        if self._optimizer_name == 'smac':
            return {
                'objective': self.evaluate_config,
                'search_space': self.search_space,
                'n_trials': 100,
                'seed': self._seed
            }
        elif self._optimizer_name == 'gp':
            return {
                'objective': self.evaluate_config,
                'search_space': self.search_space,
                'y_transform': lambda x: x,
                'iters': 100,
                'gpu': -1
            }
        elif self._optimizer_name == 'hyperband':
            return {
                'objective': self.evaluate_config,
                'search_space': self.search_space,
                'min_budget': 1,
                'max_budget': 200,
                'n_trials': 100
            }
        elif self._optimizer_name == 'pc': 
            return {
                'objective': self.evaluate_config,
                'search_space': self.search_space,
                'iterations': 20,
                'initial_samples': 5,
                'num_samples': 5,
                'use_ei': False,
                'num_self_consistency_samplings': 200,
                'max_rel_ball_size': 0.03,
                'num_ei_repeats': 10,
                'pc_type': 'parametric',
                'seed': self._seed
            }
        elif self._optimizer_name == 'einet':
            return {
                'objective': self.evaluate_config,
                'search_space': self.search_space,
                'iterations': 20,
                'init_samples': 10,
                'samples_per_iter': 5
            }
        elif self._optimizer_name == 'rs':
            return {
                'objective': self.evaluate_config,
                'search_space': self.search_space,
                'iterations': 100,
            }
        elif self._optimizer_name == 'ls':
            return {
                'objective': self.evaluate_config,
                'search_space': self.search_space,
                'runs': 100,
            }
        elif self._optimizer_name == 'optunabo':
            return {
                'objective': self.evaluate_config,
                'search_space': self.search_space,
                'iterations': 100
            }
        elif self._optimizer_name == 'skoptbo':
            return {
                'objective': self.evaluate_config,
                'search_space': self.search_space,
                'iterations': 100,
                'surrogate': 'rf'
            }
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')
        
def quadartic(x, c):
    result = -np.sum((np.array(x) + np.array(c))**2)
    return BenchQueryResult(result, result, result, cost=1)