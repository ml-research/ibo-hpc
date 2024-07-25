from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
import json

class TransNASInteractiveBenchExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, interaction_idx, task='cifar10') -> None:
        self.benchmark_name = 'transnas'
        self.benchmark_config = {
            'task': task
        }
        self._interaction_idx = interaction_idx
        benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, 
                                                        self.benchmark_config)

        self._optimizer_name = optimizer_name
        self.optimizer_config = self.get_optimizer_config(benchmark)
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)  
        super().__init__(benchmark, optimizer)      

    def run(self):
        if self._optimizer_name == 'pc':
            interventions, iterations = self.get_pc_interventions()
            self.optimizer.intervene(interventions, iterations)
        else:
            # pibo case
            intervention = self.get_pibo_intervention()
            self.optimizer.intervene(intervention)
        config, performance = self.optimizer.optimize()
        if self._optimizer_name == 'pc':
            processed_config = {}
            for name, idx in config.items():
                param_def = self.benchmark.search_space.search_space_definition[name]
                processed_config[name] = param_def['allowed'][int(idx)]
            config = processed_config
        print((config, performance))

    def evaluate_config(self, cfg, budget=None):
        test_cfg = cfg
        for key, val in cfg.items():
            cfg[key] = int(val)
        if self.benchmark is not None:
            res = self.benchmark.query(test_cfg, budget)
            return res
        
    def get_optimizer_config(self, benchmark):
        if self._optimizer_name == 'pibo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 1000
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
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')

    def get_pc_interventions(self):
        with open('./interventions/transnas_jigsaw.json', 'r') as f:
            ints_json = json.load(f)

        interaction_iters, interactions = [], []
        for interaction in ints_json:
            interaction_iters.append(interaction['iteration'])
            interactions.append(interaction['intervention'])
        
        if self._interaction_idx == -1:
            return interactions, interaction_iters
        else:
            return [interactions[self._interaction_idx]], [interaction_iters[self._interaction_idx]]

    def get_pibo_intervention(self):
        return {}