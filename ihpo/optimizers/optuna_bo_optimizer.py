from .optimizer import Optimizer
from ..search_spaces import SearchSpace
from ..benchmarks import BenchQueryResult
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
import optuna

class OptunaBOOptimizer(Optimizer):

    def __init__(self, search_space, objective, **kwargs) -> None:
        super().__init__(search_space, objective)
        self._iters = kwargs['iterations']
        if 'mode' in kwargs:
            self._mode = kwargs['mode']
        else: 
            self._mode = 'max'
        self.study = optuna.create_study()
        hp_def = self.search_space.to_configspace()
        self.cfg_space_def = {}
        self._history = []
        for hp_name in hp_def.get_all_unconditional_hyperparameters():
            hp = hp_def.get_hyperparameter(hp_name)
            if isinstance(hp, CategoricalHyperparameter):
                self.cfg_space_def[hp.name] = {'type': 'cat', 'vals': hp.choices}
            elif isinstance(hp, UniformFloatHyperparameter):
                self.cfg_space_def[hp.name] = {'type': 'float', 'low': hp.lower, 'high': hp.upper, 'is_log': hp.log}
            elif isinstance(hp, UniformIntegerHyperparameter):
                self.cfg_space_def[hp.name] = {'type': 'int', 'low': hp.lower, 'high': hp.upper, 'is_log': hp.log}
            
        
    def _objective_wrapper(self, trial):
        arg_dict = {}
        for name, hp in self.cfg_space_def.items():
            if hp['type'] == 'cat':
                arg_dict[name] = trial.suggest_categorical(name, hp['vals'])
            elif hp['type'] == 'float':
                arg_dict[name] = trial.suggest_float(name, hp['low'], hp['high'], log=hp['is_log'])
            elif hp['type'] == 'int':
                arg_dict[name] = trial.suggest_int(name, hp['low'], hp['high'], log=hp['is_log'])
        res = self.objective(arg_dict)
        self._history.append((arg_dict, res, trial.number))
        if self._mode == 'max':
            return -res.val_performance 
        else:
            return res.val_performance

    def optimize(self):
        self.study.optimize(self._objective_wrapper, n_trials=self._iters)
        return self.study.best_params, self.study.best_value
    
    @property
    def history(self):
        return self._history