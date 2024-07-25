from syne_tune.optimizer.schedulers.transfer_learning import BoundingBox
from syne_tune.optimizer.baselines import BayesianOptimization
from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import PythonBackend
from .optimizer import Optimizer
from ..search_spaces import SearchSpace

class BoundingBoxOptimizer(Optimizer):

    def __init__(self, search_space: SearchSpace, objective, 
                 transfer_learning_evaluations, max_trials=10) -> None:
        super().__init__(search_space, objective)
        sched_fun = lambda space, mode, metric: BayesianOptimization(
            space,
            'accuracy'
        )
        self.optimizer = BoundingBox(
            sched_fun,
            search_space.to_synetune(), 
            transfer_learning_evaluations=transfer_learning_evaluations,
            metric='accuracy'
        )
        self.search_space = search_space
        self.objective = objective
        self._max_trials = max_trials

    def optimize(self):
        tuner = Tuner(
            trial_backend=PythonBackend(self.objective, 
            config_space=self.search_space.to_synetune()),
            scheduler=self.optimizer,
            stop_criterion=StoppingCriterion(max_num_trials_finished=self._max_trials),
            n_workers=1
        )
        tuner.run()

        if tuner.tuning_status is not None:
            return tuner.tuning_status.get_dataframe()