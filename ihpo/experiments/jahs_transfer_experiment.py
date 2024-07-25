from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
from copy import deepcopy

class JAHSBenchExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, task1='cifar10', task2='colorectal_histology') -> None:
        self.benchmark_name = 'jahs'
        self.benchmark_config1 = {
            'task': task1
        }
        self.benchmark_config2 = {
            'task': task2
        }
        benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, 
                                                        self.benchmark_config1)

        self._optimizer_name = optimizer_name
        self.optimizer_config = self.get_optimizer_config(benchmark)
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)   
        super().__init__(benchmark, optimizer)     

    def run(self):
        # run on first task
        config, performance = self.optimizer.optimize()
        self.benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, 
                                                        self.benchmark_config2)
        # TODO: adapt search space (we need second search space definition)
        self.optimizer.adapt_to_search_space()
        # perform optimization on second search space
        config, performance = self.optimizer.optimize()
        if self._optimizer_name == 'pc':
            processed_config = {}
            for name, idx in config.items():
                param_def = self.benchmark.search_space.search_space_definition[name]
                if 'allowed' in list(param_def.keys()):
                    processed_config[name] = param_def['allowed'][int(idx)]
                else:
                    processed_config[name] = idx
            config = processed_config
        print((config, performance))

    def evaluate_config(self, cfg, budget=None):
        test_cfg = cfg
        if self._optimizer_name == 'pc':
            to_be_transformed = ['Activation', 'Op1', 'Op2', 'Op3', 'Op4', 'Op5', 'Op6', 'TrivialAugment']
            search_space_def = self.benchmark.search_space.get_search_space_definition()
            cfg_copy = deepcopy(cfg)
            for key, val in cfg.items():
                if key in to_be_transformed:
                    cfg_copy[key] = search_space_def[key]['allowed'][int(val)]
            test_cfg = cfg_copy
        if self.benchmark is not None:
            # fix fidelities
            test_cfg['Optimizer'] = 'SGD'
            test_cfg['W'] = 16
            test_cfg['N'] = 5
            test_cfg['Resolution'] = 1.0
            res = self.benchmark.query(test_cfg, budget)
            return res
        
    def get_optimizer_config(self, benchmark):
        assert benchmark is not None, 'Something went wrong instantiating the benchmark'
        if self._optimizer_name == 'smac':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 100,
                'num_iter': 2000
            }
        elif self._optimizer_name == 'hyperband':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'min_budget': 1,
                'max_budget': 100,
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
                'iterations': 2000,
            }
        elif self._optimizer_name == 'ls':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'runs': 150
            }
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')
