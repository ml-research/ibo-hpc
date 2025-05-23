from syne_tune.optimizer.schedulers.transfer_learning import BoundingBox
from syne_tune.optimizer.baselines import BayesianOptimization
from syne_tune import Tuner, StoppingCriterion
from syne_tune.config_space import Categorical
from syne_tune.optimizer.schedulers.transfer_learning import TransferLearningTaskEvaluations
from syne_tune.backend import PythonBackend
from .optimizer import Optimizer
from ..search_spaces import SearchSpace
from ..runner import SyneTuneRunner
import pandas as pd
import numpy as np

class BoundingBoxOptimizer(Optimizer):

    def __init__(self, search_space: SearchSpace, objective, 
                 transfer_learning_evaluation_files, max_trials=10, num_hyperparameters_per_task=10) -> None:
        super().__init__(search_space, objective)
        self._tfl_evals = self._prepare_prior_runs(transfer_learning_evaluation_files)
        def sched_fun(space, mode, metric): 
            # NOTE: Here we do some preprocessing.
            # syne_tune somehow convertes numbers to numpy-numbers but expects Python-numbers at some later point.
            # Fix this here.
            for k in space.keys():
                if isinstance(space[k], Categorical):
                    space[k].categories = np.array(space[k].categories).tolist()
            return BayesianOptimization(
            space,
            metric,
            search_options=dict(max_size_data_for_model=40),
            mode=mode
        )
        self.optimizer = BoundingBox(
            sched_fun,
            search_space.to_synetune(), 
            transfer_learning_evaluations=self._tfl_evals,
            metric='val_performance',
            num_hyperparameters_per_task=num_hyperparameters_per_task,
            mode='max'
        )
        self.search_space = search_space
        self.objective = objective
        self._max_trials = max_trials
        self.tuner = None

    def optimize(self):
        backend = SyneTuneRunner(self.objective)
        self.tuner = Tuner(
            trial_backend=backend,
            scheduler=self.optimizer,
            stop_criterion=StoppingCriterion(max_num_trials_finished=self._max_trials),
            n_workers=8
        )
        self.tuner.run()

        df = self.tuner.tuning_status.get_dataframe()
        return self.tuner.best_config(), df['val_performance'].max()
        
    def _prepare_prior_runs(self, task_file_dict):
        """
            Prepare the prior HPO runs.
            Assumes files to be given as a list of csv files which will be converted into Evaluations.
        """
        hyperparameter_names = list(self.search_space.get_search_space_definition().keys())
        config_sapce = self.search_space.to_synetune()
        task_to_evals_dict = {}
        for task, files in task_file_dict.items():
            hyperparameter_values = []
            evaluation_scores = []

            for f in files:
                df = pd.read_csv(f, index_col=0)

                configs = df[hyperparameter_names]
                scores = df['val_performance']

                hyperparameter_values.append(configs)
                evaluation_scores.append(scores)

            hyperparameter_values = pd.concat(hyperparameter_values)
            evaluation_scores = np.array(evaluation_scores).flatten().reshape(-1, len(files), 1, 1)

            try:
                tfl_evals = TransferLearningTaskEvaluations(config_sapce, hyperparameter_values, ['val_performance'], evaluation_scores)
                task_to_evals_dict[task] = tfl_evals
            except AssertionError:
                raise RuntimeError('BoundingBoxOptimizer can only handle one prior run per task')

        return task_to_evals_dict

    @property
    def history(self):
        if self.tuner is None:
            raise RuntimeError('History only accessable after performing optimization')
        if self.tuner.tuning_status is not None:
            result_df = self.tuner.tuning_status.get_dataframe()
            history = []
            hyperparam_names = list(self.search_space.get_search_space_definition().keys())
            for i, data in result_df.iterrows():
                cfg = data[hyperparam_names].to_dict()
                res = self.objective(cfg)
                history.append((cfg, res, i))
            return history