from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
import json
from ConfigSpace import CategoricalHyperparameter
import os
import numpy as np

class NASBench201InteractiveExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, interaction_idx, task='cifar10') -> None:
        self.benchmark_name = 'nas201'
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
        # register intervention
        if self._optimizer_name == 'pc':
            interventions, iterations = self.get_pc_interventions()
            self.optimizer.intervene(interventions, iterations)
        elif self._optimizer_name == 'bopro':
            interventions = self.get_bopro_intervention()
            self.optimizer.intervene(interventions)
        else:
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
        assert benchmark is not None, 'Something went wrong instantiating the benchmark'
        if self._optimizer_name == 'pibo':
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
        elif self._optimizer_name == 'pc': 
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 100,
                'samples_per_iter': 20,
                'use_eic': False,
                'eic_samplings': 20,
                'interaction_dist_sample_decay': 0.9,
            }
        elif self._optimizer_name == 'pibo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 2000,
            }
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')
        
    def get_pc_interventions(self):

        with open('./interventions/nas201_cifar10.json', 'r') as f:
            ints_json = json.load(f)
        
        interaction_iters, interactions = [], []
        for interaction in ints_json:
            interaction_iters.append(interaction['iteration'])
            interactions.append(interaction['intervention'])
        
        if len(self._interaction_idx) == 1:
            idx = self._interaction_idx[0]
            if idx == -1:
                return interactions, interaction_iters
            else:
                return [interactions[idx]], [interaction_iters[idx]]
        else:
            final_interactions, final_iterations = [], []
            for i in self._interaction_idx:
                final_interactions.append(interactions[i])
                final_iterations.append(interaction_iters[i])
            return final_interactions, final_iterations
        
    def get_pibo_intervention(self):
        return {
            "Op_0": CategoricalHyperparameter("Op_0", list(range(5)), weights=[1, 1, 1e4, 1, 1]), 
            "Op_1": CategoricalHyperparameter("Op_1", list(range(5)), weights=[1, 1, 1e4, 1, 1]), 
            "Op_2": CategoricalHyperparameter("Op_2", list(range(5)), weights=[1e4, 1, 1, 1, 1])
            }

    def get_bopro_intervention(self):
        op_1_prior = np.array([1, 1, 1000, 1, 1]) / sum([1, 1, 1000, 1, 1])
        op_2_prior = np.array([1, 1, 1000, 1, 1]) / sum([1, 1, 1000, 1, 1])
        op_3_prior = np.array([1000, 1, 1, 1, 1]) / sum([1000, 1, 1, 1, 1])
        return {
            "Op_0": {"prior": op_1_prior.tolist()}, 
            "Op_1": {"prior": op_2_prior.tolist()}, 
            "Op_2": {"prior": op_3_prior.tolist()}
            }
    
    def setup_bopro_json(self, benchmark):
        borpo_search_space = benchmark.search_space.to_hypermapper()
        json_dict = {
            "application_name": "NAS201 Interactive",
            "optimization_objectives": ["value"],
            "design_of_experiment": {
                "number_of_samples": 3,
            },
            "optimization_iterations": 2000,
            "acquisition_function": "EI",
            "number_of_cpus": 16,
            "optimization_method": "prior_guided_optimization",
            "models": {
                "model": "random_forest"
            },
            "input_parameters": borpo_search_space
        }
        if not os.path.exists('./bopro_experiments/'):
            os.mkdir('./bopro_experiments')
        self._bopro_file_name =  './bopro_experiments/nas201_interactive.json'
        with open(self._bopro_file_name, 'w+') as f:
            json.dump(json_dict, f)