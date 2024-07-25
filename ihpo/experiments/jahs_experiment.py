from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
from copy import deepcopy

class JAHSBenchExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, task='cifar10') -> None:
        self.benchmark_name = 'jahs'
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
        test_cfg = deepcopy(cfg)
        if self._optimizer_name == 'pc':
            to_be_transformed = ['Activation', 'W', 'N', 'epoch', 'Op1', 'Op2', 'Op3', 'Op4', 'Op5', 'Op6', 'TrivialAugment']
            search_space_def = self.benchmark.search_space.get_search_space_definition()
            cfg_copy = deepcopy(cfg)
            for key, val in cfg.items():
                if key in to_be_transformed:
                    cfg_copy[key] = search_space_def[key]['allowed'][int(val)]
            test_cfg = cfg_copy
        if self.benchmark is not None:
            test_cfg['Optimizer'] = 'SGD'
            if budget is None and 'epoch' in cfg:
                budget = cfg['epoch']
                test_cfg.pop('epoch', None)
            res = self.benchmark.query(test_cfg, budget)
            return res
        
    def get_optimizer_config(self, benchmark):
        assert benchmark is not None, 'Something went wrong instantiating the benchmark'
        if self._optimizer_name == 'smac':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 2000
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
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')
