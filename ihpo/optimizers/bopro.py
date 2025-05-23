from typing import Dict, List
from .optimizer import Optimizer
from ..search_spaces import SearchSpace
import numpy as np
from hypermapper import optimizer
import json

class BOPrOOptimizer(Optimizer):

    def __init__(self, search_space: SearchSpace, objective, exp_json):
        self.search_space = search_space
        self.objective = objective
        self.evaluations = []
        self.exp_file = exp_json

    def optimize(self):
        def _objective_wrapper(cfg, budget=None):
            res = self.objective(cfg, budget=budget)
            if res is None:
                return 0.0
            self.evaluations.append((cfg, res, len(self.evaluations)))
            return -res.val_performance
        optimizer.optimize(self.exp_file, _objective_wrapper)
        eval_results = [res.test_performance for _, res, _ in self.evaluations]
        max_idx = eval_results.index(max(eval_results))
        config, res, _ = self.evaluations[max_idx]
        return config, res

    def intervene(self, intervention):
        with open(self.exp_file, 'r') as f:
            bopro_json = json.load(f)
        input_params = bopro_json['input_parameters']
        for key, val in intervention.items():
            input_param = input_params[key]
            input_param = {**input_param, **val}
            input_params[key] = input_param
        bopro_json['input_parameters'] = input_params
        with open(self.exp_file, 'w+') as f:
            json.dump(bopro_json, f)

    @property
    def history(self):
        return self.evaluations
