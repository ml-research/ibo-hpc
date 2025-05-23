from ..search_spaces import SearchSpace
from ..benchmarks import BenchQueryResult
from .optimizer import Optimizer
from typing import List, Dict
import numpy as np

class RandomSearchOptimizer(Optimizer):

    def __init__(self, search_space: SearchSpace, objective, iterations=100, 
                 interventions=None, intervention_iters=None) -> None:
        super().__init__(search_space, objective)
        self.search_space = search_space
        self.objective = objective
        self._trial_id = 0
        self._iterations = iterations
        self._interventions = interventions
        self._intervention_iters = intervention_iters
        self._curr_intervention = -1
        
    def optimize(self):
        self._trial_id += 1
        configs = self.search_space.sample(size=self._iterations)
        for i, config in enumerate(configs):
            if self._interventions is not None:# and self._intervention_duration is not None:
                if i in self._intervention_iters:
                    self._curr_intervention = self._intervention_iters.index(i)
                    print(f"[Optimizer][RS] Using Intervention {self._curr_intervention}: {self._interventions[self._curr_intervention]}")
                if self._curr_intervention > -1:
                    config = self._apply_intervention(config, self._interventions[self._curr_intervention])
            score: BenchQueryResult = self.objective(config)
            self.evaluations.append((config, score, i))
        scores = [s.val_performance for _, s, _ in self.evaluations]
        best_idx = scores.index(max(scores))
        return self.evaluations[best_idx][:2]
    
    def intervene(self, interventions: List[Dict], iters):
        """
            Set certain dimensions to a fixed value to obtain a conditional distribution
        """
        self._interventions = interventions
        self._intervention_iters = iters

    def _apply_intervention(self, config, intervention):
        ssd = self.search_space.get_search_space_definition()

        def sample_from_condition_prior(intervention, intervention_key):
            if intervention[intervention_key]['dist'] == 'gauss':
                mu, std = intervention[intervention_key]['parameters']
                samples = np.random.normal(mu, std, 1).item()
            elif intervention[intervention_key]['dist'] == 'uniform':
                s, e = intervention[intervention_key]['parameters']
                samples = np.random.uniform(s, e, 1).item()
            elif intervention[intervention_key]['dist'] == 'int_uniform':
                s, e = intervention[intervention_key]['parameters']
                samples = np.random.randint(s, e, 1).item()
            elif intervention[intervention_key]['dist'] == 'cat':
                weights = intervention[intervention_key]['parameters']
                probs = np.array(weights) / np.sum(weights)
                idx = np.random.choice(np.arange(len(probs)), 1, p=probs).item()
                samples = ssd[intervention_key]['allowed'][idx]
            return samples
        
        # get configuration and replace HPs defined in intervention.
        for key in config.keys():
            if key in intervention:
                if isinstance(intervention[key], dict):
                    config[key] = sample_from_condition_prior(intervention, key)
                else:
                    config[key] = intervention[key]

        return config
    
    @property
    def history(self):
        return self.evaluations
