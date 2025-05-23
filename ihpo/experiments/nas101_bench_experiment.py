from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory, BenchQueryResult
from ..optimizers import OptimizerFactory
import numpy as np
from copy import deepcopy

class NASBench101Experiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, task='cifar10', handle_invalid_configs=False, seed=0) -> None:
        self.benchmark_name = 'nas101'
        self.benchmark_config = {
            'task': task
        }
        benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, 
                                                        self.benchmark_config)

        self._optimizer_name = optimizer_name
        self.optimizer_config = self.get_optimizer_config(benchmark)
        self._seed = seed
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)   
        self._handle_invalid_configs = handle_invalid_configs
        super().__init__(benchmark, optimizer)     

    def run(self):
        config, performance = self.optimizer.optimize()
        print((config, performance))

    def evaluate_config(self, cfg, budget=None):
        test_cfg = cfg
        if budget is not None:
            budget = self._get_closest_budget(budget)
        if self.benchmark is not None:
            res = self.benchmark.query(test_cfg, budget)
            return res
        elif self._handle_invalid_configs:
            return BenchQueryResult(10.06, 10.06, 10.06, cost=1206.45, epochs=108)
        
    def get_optimizer_config(self, benchmark):
        assert benchmark is not None, 'Something went wrong instantiating the benchmark'
        if self._optimizer_name == 'smac':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 2000,
                'log_hpo_runtime': True,
                'seed': self._seed
            }
        elif self._optimizer_name == 'hyperband':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'min_budget': 1,
                'max_budget': 110,
                'n_trials': 500
            }
        elif self._optimizer_name == 'pc': 
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 100,
                'num_samples': 20,
                'use_ei': False,
                'num_ei_repeats': 20,
                'pc_type': 'quantile'
            }
        elif self._optimizer_name == 'rs':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 2000,
            }
        elif self._optimizer_name == 'ls':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'runs': 150
            }
        elif self._optimizer_name == 'pibo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 2000,
            }
        elif self._optimizer_name == 'optunabo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 2000
            }
        elif self._optimizer_name == 'skoptbo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 2000,
                'surrogate': 'rf'
            }
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')
        
    def _get_closest_budget(self, budget):
        allowed = np.array([4, 12, 36, 108])
        idx = abs(allowed - budget).argmin()
        return allowed[idx]
