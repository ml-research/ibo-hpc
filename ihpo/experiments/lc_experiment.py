from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
import warnings

class LCExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, task='adult', seed=0) -> None:
        self.benchmark_name = 'lcbench'
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
            warnings.warn('LCBench does not support multiple fidelities.')
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
                'num_self_consistency_samplings': 50,
                'max_rel_ball_size': [0.05, 0.1, 0.1, 0.01, 0.02, 0.1, 0.001],
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
