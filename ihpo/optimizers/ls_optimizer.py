from .optimizer import Optimizer
import numpy as np
from ..search_spaces import SearchSpace

class LocalSearchOptimizer(Optimizer):

    def __init__(self, search_space: SearchSpace, objective, runs=10) -> None:
        super().__init__(search_space, objective)
        self.objective = objective
        self.search_space = search_space
        self._runs = runs
        
    def optimize(self):
        """
            Implements local search.
            candidate_generator is a function that suggests new candidates that 
            should be evaluated (e.g. all neighbours of a given candidate).
        """
        best_candidate = None
        best_score = None
        self.hist = []
        for i in range(self._runs):
            if i % 50 == 0:
                print(f"Iteration {i}/{self._runs}")
            candidates = self.search_space.get_neighbors(best_candidate)
            scores = []

            for c in candidates:
                obj = self.objective(c)
                scores.append(obj)
            
            temp_hist = [(c, s, i) for c, s in zip(candidates, scores)]
            self.hist += temp_hist

            val_scores = [s.val_performance for s in scores]
            best_score_idx = np.argmax(val_scores).flatten()[0]
            best_candidate = candidates[best_score_idx]
            best_score = scores[best_score_idx]
        return best_candidate, best_score
    
    @property
    def history(self):
        return self.hist
