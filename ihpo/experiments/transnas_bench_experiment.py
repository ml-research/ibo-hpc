from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
from copy import deepcopy

class TransNASBenchExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, task='cifar10', seed=0) -> None:
        self.benchmark_name = 'transnas'
        self.benchmark_config = {
            'task': task
        }
        benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, 
                                                        self.benchmark_config)
        self._optimizer_name = optimizer_name
        self.optimizer_config = self.get_optimizer_config(benchmark)
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)  
        super().__init__(benchmark, optimizer)      

    def run(self):
        config, performance = self.optimizer.optimize()
        print((config, performance))

    def evaluate_config(self, cfg, budget=None):
        #test_cfg = cfg
        #print(cfg)
        #if self._optimizer_name == 'pc':
        #    processed_config = {}
        #    for name, idx in cfg.items():
        #        param_def = self.benchmark.search_space.search_space_definition[name]
        #        processed_config[name] = param_def['allowed'][int(idx)]
        #    test_cfg = processed_config
        #print(test_cfg)
        if self.benchmark is not None:
            res = self.benchmark.query(cfg, budget)
            return res
        
    def get_optimizer_config(self, benchmark):
        if self._optimizer_name == 'smac':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 2000,
                'log_hpo_runtime': True
            }
        elif self._optimizer_name == 'pibo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 2000
            }
        elif self._optimizer_name == 'bopro':
            self.setup_bopro_json(benchmark)
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'exp_json': self._bopro_file_name
            }
        elif self._optimizer_name == 'hyperband':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'min_budget': 1,
                'max_budget': 200,
                'n_trials': 2000
            }
        elif self._optimizer_name == 'pc':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 100,
                'samples_per_iter': 20,
                'use_eic': False,
                'eic_samplings': 20,
                'log_hpo_runtime': True
                #'conditioning_value_quantile': 0.25
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
