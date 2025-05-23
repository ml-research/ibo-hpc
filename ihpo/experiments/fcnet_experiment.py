from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
import warnings

class FCNetExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, task='naval', seed=0) -> None:
        self.benchmark_name = 'fcnet'
        self.benchmark_config = {
            'task': task
        }
        benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, 
                                                        self.benchmark_config)

        self._optimizer_name = optimizer_name
        self._seed = seed
        self.optimizer_config = self.get_optimizer_config(benchmark)
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)   
        super().__init__(benchmark, optimizer)     

    def run(self):
        config, performance = self.optimizer.optimize()
        print((config, performance))

    def evaluate_config(self, cfg, budget=None):
        test_cfg = cfg
        if budget is not None:
            warnings.warn('FCNet does not support multiple fidelities.')
        res = self.benchmark.query(test_cfg, budget)
        return res
        
    def get_optimizer_config(self, benchmark):
        assert benchmark is not None, 'Something went wrong instantiating the benchmark'
        if self._optimizer_name == 'smac':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 100,
                'log_hpo_runtime': True,
                'seed': self._seed
            }
        elif self._optimizer_name == 'pc': 
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 20,
                'num_samples': 5,
                'initial_samples': 10,
                'num_self_consistency_samplings': 20,
                'max_rel_ball_size': [0, 0, 0, 0, 0, 0.2, 0, 0.2, 0.2],
                'use_ei': False,
                'num_ei_repeats': 20,
                'pc_type': 'mspn'
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
                'runs': 200
            }
        elif self._optimizer_name == 'optunabo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 100
            }
        elif self._optimizer_name == 'skoptbo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 100,
                'surrogate': 'rf'
            }
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')
