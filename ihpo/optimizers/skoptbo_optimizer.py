from .optimizer import Optimizer
from ..search_spaces import SearchSpace
from ..benchmarks import BenchQueryResult
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize, forest_minimize
from skopt.utils import use_named_args
import numpy as np

class SkOptBOOptimizer(Optimizer):

    def __init__(self, search_space, objective, **kwargs) -> None:
        super().__init__(search_space, objective)
        self._history = []
        self._iter = 0
        self._iterations = kwargs['iterations']
        self._surrogate = kwargs['surrogate']
        self.space = []
        hp_def = search_space.to_configspace()
        for hp_name in hp_def.get_all_unconditional_hyperparameters():
            hp = hp_def.get_hyperparameter(hp_name)
            if isinstance(hp, CategoricalHyperparameter):
                skopt_hp = Categorical(hp.choices, name=hp.name)
                self.space.append(skopt_hp)
            elif isinstance(hp, UniformFloatHyperparameter):
                prior = 'uniform' if not hp.log else 'log-uniform'
                skopt_hp = Real(hp.lower, hp.upper, name=hp.name, prior=prior)
                self.space.append(skopt_hp)
            elif isinstance(hp, UniformIntegerHyperparameter):
                prior = 'uniform' if not hp.log else 'log-uniform'
                skopt_hp = Integer(hp.lower, hp.upper, name=hp.name, prior=prior)
                self.space.append(skopt_hp)

    def optimize(self):
        
        @use_named_args(self.space)
        def _objective(**params):
            if self._iter % 10 == 0:
                print(f"Iteration {self._iter}/{self._iterations}")
            res = self.objective(params)
            self._history.append((params, res, self._iter))
            self._iter += 1
            return -res.val_performance
        
        if self._surrogate == 'gp':
            res = gp_minimize(_objective, self.space, n_calls=self._iterations)
        elif self._surrogate == 'rf':
            res = forest_minimize(_objective, self.space, n_calls=self._iterations, n_random_starts=5)
        else:
            raise ValueError(f'No such surrogate: {self._surrogate}')
        scores = [r.test_performance for _, r, _ in self._history]
        best_idx = np.argmax(scores).flatten()[0]
        best = self._history[best_idx]
        return best[0], scores[best_idx]

    @property
    def history(self):
        return self._history