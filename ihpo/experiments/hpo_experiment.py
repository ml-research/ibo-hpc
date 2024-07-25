from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
import pandas as pd

class HPOExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, task=167120) -> None:
        self.benchmark_config = {
            'model': 'xgb',
            'task_id': int(task)
        }
        benchmark = BenchmarkFactory.get_benchmark('hpo', self.benchmark_config)
        self._optimizer_name = optimizer_name   
        self.optimizer_config = self.get_optimizer_config(benchmark)
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)        
        super().__init__(benchmark, optimizer)

    def run(self):
        config, performance = self.optimizer.optimize()
        if self._optimizer_name == 'pc':
            processed_config = {}
            for name, idx in config.items():
                param_def = self.benchmark.search_space.search_space_definition[name]
                processed_config[name] = param_def['allowed'][int(idx)]
            config = processed_config
        print((config, performance))

    def evaluate_config(self, cfg):
        # if PC is optimizer, we have to discretize search space due to tabular benchmark
        # -> convert discretized configs back to original domain
        if self._optimizer_name == 'pc' and self.benchmark is not None:
            processed_config = {}
            for name, idx in cfg.items():
                param_def = self.benchmark.search_space.search_space_definition[name]
                processed_config[name] = param_def['allowed'][int(idx)]
        else:
            processed_config = cfg
        if self.benchmark is not None:
            res = self.benchmark.query(processed_config)
            return res

    def get_optimizer_config(self, benchmark):
        if self._optimizer_name == 'smac':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 100
            }
        elif self._optimizer_name == 'hyperband':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'min_budget': 1,
                'max_budget': 200,
                'n_trials': 100
            }
        elif self._optimizer_name == 'pc': 
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 100,
                'samples_per_iter': 20,
                'use_eic': False,
                'eic_samplings': 20
            }
        elif self._optimizer_name == 'rs':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 100,
            }
        elif self._optimizer_name == 'ls':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'runs': 100,
            }
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')