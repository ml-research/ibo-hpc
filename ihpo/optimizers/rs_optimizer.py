from ..search_spaces import SearchSpace
from ..benchmarks import BenchQueryResult
from .optimizer import Optimizer

class RandomSearchOptimizer(Optimizer):

    def __init__(self, search_space: SearchSpace, objective, iterations=100) -> None:
        super().__init__(search_space, objective)
        self.search_space = search_space
        self.objective = objective
        self._trial_id = 0
        self._iterations = iterations
        
    def optimize(self):
        self._trial_id += 1
        configs = self.search_space.sample(size=self._iterations)
        for i, config in enumerate(configs):
            score: BenchQueryResult = self.objective(config)
            self.evaluations.append((config, score, i))
        scores = [s.val_performance for _, s, _ in self.evaluations]
        best_idx = scores.index(max(scores))
        return self.evaluations[best_idx][:2]
    
    @property
    def history(self):
        return self.evaluations
