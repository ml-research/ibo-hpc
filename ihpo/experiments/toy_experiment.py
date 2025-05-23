from .experiment import BenchmarkExperiment
from ..optimizers import OptimizerFactory
from ..search_spaces import BraninSearchSpace, RealSearchSpace
from ..benchmarks import BenchQueryResult
import numpy as np

class ToyExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, function='branin', rb_dims=2, seed=0) -> None:
        benchmark = None
        self._function = function
        self._seed = seed
        self._optimizer_name = optimizer_name
        if function == 'branin': 
            self.search_space = BraninSearchSpace()
        elif function == 'rosenbrock':
            self.search_space = RealSearchSpace(rb_dims)
        else:
            raise ValueError(f'No such function: {function}')
        self.optimizer_config = self.get_optimizer_config()
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)
        super().__init__(benchmark, optimizer)

    def run(self):
        config, performance = self.optimizer.optimize()
        print((config, performance))

    def evaluate_config(self, cfg):
        if self._function == 'branin':
            return branin(cfg['x_1'], cfg['x_2'])
        elif self._function == 'rosenbrock':
            config_vec = list(cfg.values())
            return rosenbrock(config_vec)

    def get_optimizer_config(self):
        if self._optimizer_name == 'smac':
            return {
                'objective': self.evaluate_config,
                'search_space': self.search_space,
                'n_trials': 2000,
                'seed': self._seed
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
                'iterations': 10,
                'num_samples': 10,
                'use_ei': False,
                'num_self_consistency_samplings': 1000,
                'num_ei_repeats': 10,
                'pc_type': 'parametric',
                'seed': self._seed
            }
        elif self._optimizer_name == 'rs':
            return {
                'objective': self.evaluate_config,
                'search_space': self.search_space,
                'iterations': 2000,
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
                'iterations': 2000
            }
        elif self._optimizer_name == 'skoptbo':
            return {
                'objective': self.evaluate_config,
                'search_space': self.search_space,
                'iterations': 1000,
                'surrogate': 'rf'
            }
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')
        
def branin(x1, x2):
    a = 1
    b = 5.1 / (4 * np.pi)**2
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    obj = a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*np.cos(x1) + s
    return BenchQueryResult(obj, obj, obj, cost=1)

def rosenbrock(x):
    result = -sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))
    return BenchQueryResult(result, result, result, cost=1)