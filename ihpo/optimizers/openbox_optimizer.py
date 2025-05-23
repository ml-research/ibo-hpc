from .optimizer import Optimizer
from openbox import Observation, History, Advisor, space as sp, logger
from ConfigSpace import Configuration
import numpy as np
import pandas as pd


class OpenBoxOptimizer(Optimizer):

    def __init__(self, search_space, objective, 
                 transfer_learning_evaluation_files, method='tlbo_rgpe_prf', seed=0, max_iter=2000) -> None:
        self.search_space = search_space
        transfer_learning_history = self._prepare_prior_runs(transfer_learning_evaluation_files)
        self._max_iter = max_iter
        self._history = []
        cfg_space = search_space.to_configspace()
        self.advisor = Advisor(
            config_space=cfg_space,
            num_objectives=1,
            num_constraints=0,
            initial_trials=3,
            transfer_learning_history=transfer_learning_history,  # type: List[History]
            surrogate_type=method,
            acq_type='ei',
            acq_optimizer_type='local_random',
            task_id='TLBO',
        )
        super().__init__(search_space, objective, seed)

    def optimize(self):
        for i in range(self._max_iter):
            if i % 5 == 0:
                print(f"Iteration {i + 1}/{self._max_iter}")
            config = self.advisor.get_suggestion()
            res = self.objective(config.get_dictionary())
            observation = Observation(config=config, objectives=-res.val_performance)
            self.advisor.update_observation(observation)
            self._history.append((config, res, i))

        results = [r.val_performance for _, r, _ in self._history]
        argmax = np.argmax(results).flatten()[0]
        return self._history[argmax][:2]

    def _prepare_prior_runs(self, task_file_dict):
        """
            Prepare the prior HPO runs.
            Assumes files to be given as a list of csv files which will be converted into Evaluations.
        """
        hyperparameter_names = list(self.search_space.get_search_space_definition().keys())
        config_space = self.search_space.to_configspace()
        histories = []
        print("Load prior runs...")
        for task, files in task_file_dict.items():
            print(f"Loading task {task}")
            history = History(task_id=task, config_space=config_space)

            for f in files:
                df = pd.read_csv(f, index_col=0)

                configs = df[hyperparameter_names]
                scores = df['test_performance']

                for (_, cfg), score in zip(configs.iterrows(), scores):
                    configuration = Configuration(config_space, cfg.to_dict())
                    obs = Observation(config=configuration, objectives=[score])
                    history.update_observation(obs)

            histories.append(history)
        print("Done loading")
        return histories
    
    @property
    def history(self):
        return self._history
