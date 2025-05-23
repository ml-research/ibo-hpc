from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
import os
import numpy as np
import json
from scipy.special import softmax
from ConfigSpace import CategoricalHyperparameter, UniformFloatHyperparameter, NormalFloatHyperparameter, UniformIntegerHyperparameter
import warnings

class LCInteractiveExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, task='adult', interaction_idx=0, seed=0) -> None:
        self.benchmark_name = 'lcbench'
        self.benchmark_config = {
            'task': task
        }
        benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, 
                                                        self.benchmark_config)

        self._optimizer_name = optimizer_name
        self._seed = seed
        self._interaction_idx = interaction_idx
        self.optimizer_config = self.get_optimizer_config(benchmark)
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)   
        super().__init__(benchmark, optimizer)     

    def run(self):
        if self._optimizer_name == 'pc' or self._optimizer_name == 'rs':
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
                'log_hpo_runtime': False,
                'seed': self._seed
            }
        elif self._optimizer_name == 'pc': 
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 10,
                'num_samples': 10,
                'num_self_consistency_samplings': 10,
                'initial_samples': 10,
                'use_ei': False,
                'num_ei_repeats': 20,
                'pc_type': 'mspn',
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
                'runs': 150
            }
        if self._optimizer_name == 'pibo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 100,
                'seed': self._seed
            }
        elif self._optimizer_name == 'bopro':
            self.setup_bopro_json(benchmark)
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'exp_json': self._bopro_file_name
            }
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')
        

    def get_pc_interventions(self):
        with open('./interventions/lc.json', 'r') as f:
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
        print("Getting PiBO interaction. Iteration will be set to 0 by default.")
        with open('./interventions/lc.json', 'r') as f:
            ints_json = list(json.load(f))
        interaction = ints_json[self._interaction_idx[0]]
        if interaction['kind'] != 'dist':
            raise ValueError('PiBO only supports distributions as user input!')
        interaction_spec = interaction['intervention']
        interaction_dict = {}
        for k, v in interaction_spec.items():
            if v['dist'] == 'cat':
                weights = v['parameters']
                weights = softmax(weights)
                hp = CategoricalHyperparameter(k, v['values'], weights=weights)
            elif v['dist'] == 'uniform':
                mi, ma = v['parameters']
                hp = UniformFloatHyperparameter(k, mi, ma)
            elif v['dist'] == 'int_uniform':
                mi, ma = v['parameters']
                hp = UniformIntegerHyperparameter(k, mi, ma, default_value=mi)
            elif v['dist'] == 'gauss':
                mean, std = v['parameters']
                hp = NormalFloatHyperparameter(k, mean, std)
            
            interaction_dict[k] = hp
        
        return interaction_dict

    def get_bopro_intervention(self):
        print("Getting BOPrO interaction. Iteration will be set to 0 by default.")
        with open('./interventions/lc.json', 'r') as f:
            ints_json = list(json.load(f))
        interaction = ints_json[self._interaction_idx[0]]
        if interaction['kind'] != 'dist':
            raise ValueError('BOPrO only supports distributions as user input!')
        interaction_spec = interaction['intervention']
        interaction_dict = {}
        for k, v in interaction_spec.items():
            if v['dist'] == 'cat':
                weights = v['parameters']
                weights = list(softmax(weights))
                hp = {'prior': weights}
            elif v['dist'] == 'uniform' or v['dist'] == 'int_uniform':
                mi, ma = v['parameters']
                hp = {'prior': 'uniform', 'values': [mi, ma]}
            elif v['dist'] == 'gauss':
                # TODO: Currently, parameters are set in search space. Should be done here
                mean, std = v['parameters']
                hp = {'prior': 'custom_gaussian', 'custom_gaussian_prior_means': [mean], 'custom_gaussian_prior_stds': [std]}
            
            interaction_dict[k] = hp
        
        return interaction_dict
    
    def setup_bopro_json(self, benchmark):
        search_space = benchmark.search_space
        borpo_search_space = search_space.to_hypermapper()
        json_dict = {
            "application_name": "HPOB Interactive",
            "optimization_objectives": ["value"],
            "design_of_experiment": {
                "number_of_samples": 3,
            },
            "optimization_iterations": 100,
            "optimization_method": "prior_guided_optimization",
            "number_of_cpus": 0,
            "models": {
                "model": "random_forest"
            },
            "input_parameters": borpo_search_space,
            "local_search_starting_points": 10,
            "local_search_random_points": 50,
            "local_search_evaluation_limit": 200
        }
        if not os.path.exists('./bopro_experiments/'):
            os.mkdir('./bopro_experiments')
        self._bopro_file_name =  './bopro_experiments/lc_interactive.json'
        with open(self._bopro_file_name, 'w+') as f:
            json.dump(json_dict, f)
