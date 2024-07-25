from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
import json
from copy import deepcopy
from ConfigSpace import CategoricalHyperparameter, UniformFloatHyperparameter, NormalFloatHyperparameter
import os
import numpy as np

class JAHSBenchInteractiveExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, interaction_idx, task='cifar10') -> None:
        self.benchmark_name = 'jahs'
        self.benchmark_config = {
            'task': task
        }
        benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, 
                                                        self.benchmark_config)
        self._interaction_idx = interaction_idx
        self._optimizer_name = optimizer_name
        self.optimizer_config = self.get_optimizer_config(benchmark)
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)   
        super().__init__(benchmark, optimizer)     

    def run(self):
        if self._optimizer_name == 'pc':
            interventions, iterations = self.get_pc_interventions()
            self.optimizer.intervene(interventions, iterations)
        elif self._optimizer_name == 'bopro':
            intervention = self.get_bopro_intervention()
            self.optimizer.intervene(intervention)
        else:
            # pibo case
            intervention = self.get_pibo_intervention()
            self.optimizer.intervene(intervention)
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
            to_be_transformed = ['Activation', 'W', 'N', 'Op1', 'Op2', 'Op3', 'Op4', 'Op5', 'Op6', 'TrivialAugment']
            search_space_def = self.benchmark.search_space.get_search_space_definition()
            cfg_copy = deepcopy(cfg)
            for key, val in cfg.items():
                if key in to_be_transformed:
                    cfg_copy[key] = search_space_def[key]['allowed'][int(val)]
            test_cfg = cfg_copy
        elif self._optimizer_name == 'bopro':
            cfg_copy = deepcopy(cfg)
            for key, val in cfg.items():
                if key == 'TrivialAugment':
                    transformed_val = True if val == 1 else False
                    cfg_copy[key] = transformed_val
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
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')
        
    def get_pc_interventions(self):
        with open('./interventions/jahs_cifar10.json', 'r') as f:
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
        n_weights = [1] * len(list(self.benchmark.search_space.config_space['N']))
        n_weights[3] = 1e4
        w_weights = [1] * len(list(self.benchmark.search_space.config_space['W']))
        w_weights[16] = 1e4
        return {
                "N": CategoricalHyperparameter('N', list(range(16)), weights=n_weights),
                "W": CategoricalHyperparameter('W', list(range(32)), weights=w_weights),
                "Resolution": UniformFloatHyperparameter('Resolution', 0.98, 1.02)
                }

    def get_bopro_intervention(self):
        n_weights = [1] * len(list(self.benchmark.search_space.config_space['N']))
        n_weights[3] = 1e4
        n_weights = list(np.array(n_weights) / sum(n_weights))
        w_weights = [1] * len(list(self.benchmark.search_space.config_space['W']))
        w_weights[16] = 1e4
        w_weights = list(np.array(w_weights) / sum(w_weights))
        return {
                "N": {
                    "prior": n_weights
                },
                "W": {
                    "prior": w_weights
                },
                "Resolution": {
                    "prior": "custom_gaussian" # distribution automatically defined between 0.98 and 1.02 if this interval is set in search space
                }
        }
    
    def setup_bopro_json(self, benchmark):
        search_space = benchmark.search_space
        search_space.change_search_space('Resolution', [0., 1.3]) # set intervention. must be done here already for JSON file.
        borpo_search_space = search_space.to_hypermapper()
        borpo_search_space['Resolution']['prior'] = 'custom_gaussian'
        borpo_search_space['Resolution']['custom_gaussian_prior_means'] = [1.]
        borpo_search_space['Resolution']['custom_gaussian_prior_stds'] = [0.3]
        json_dict = {
            "application_name": "JAHS Interactive",
            "optimization_objectives": ["value"],
            "design_of_experiment": {
                "number_of_samples": 3,
            },
            "optimization_iterations": 2000,
            "optimization_method": "prior_guided_optimization",
            "number_of_cpus": 16,
            "models": {
                "model": "random_forest"
            },
            "input_parameters": borpo_search_space
        }
        if not os.path.exists('./bopro_experiments/'):
            os.mkdir('./bopro_experiments')
        self._bopro_file_name =  './bopro_experiments/jahs_interactive.json'
        with open(self._bopro_file_name, 'w+') as f:
            json.dump(json_dict, f)