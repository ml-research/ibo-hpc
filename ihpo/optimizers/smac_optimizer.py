from .optimizer import Optimizer
from smac import Scenario, HyperparameterOptimizationFacade
from ..search_spaces import SearchSpace
from ..benchmarks import BenchQueryResult
from ConfigSpace import Configuration
from smac.runhistory.dataclasses import TrialValue
from smac.main.config_selector import ConfigSelector
from ..utils.smac import SafeLocalAndSrotedRandomSearch

class SMACOptimizer(Optimizer):

    def __init__(self, search_space: SearchSpace, objective, 
                 cfg_selector_retries=16, **scenario_args) -> None:
        super().__init__(search_space, objective)
        self._num_iter = scenario_args['n_trials']
        self.scenario = Scenario(search_space.to_configspace(), **scenario_args)
        self.objective = objective
        maximizer = SafeLocalAndSrotedRandomSearch(
            search_space
        )
        intensifier = HyperparameterOptimizationFacade.get_intensifier(
            self.scenario,
            max_config_calls=1,  # We basically use one seed per config only
        )
        self.smac = HyperparameterOptimizationFacade(
            self.scenario,
            self._objective_wrapper,
            acquisition_maximizer=maximizer,
            overwrite=True,
            config_selector=ConfigSelector(
                self.scenario, retries=cfg_selector_retries
            ),
            intensifier=intensifier
        )
        self.search_space = search_space
        self.hist = []

    def _objective_wrapper(self, config: Configuration, 
                           seed) -> float:
        res: BenchQueryResult = self.objective(config)
        return res.val_performance

    def optimize(self):
        """
            Perform one step of SMAC.
        """
        evaluations, configs = [], []
        for i in range(self._num_iter):
            if i % 50 == 0:
                print(f"Iteration {i}/{self._num_iter}")
            try:
                info = self.smac.ask()

                if self.search_space.is_valid(info.config):
                    # only evaluate valid configs
                    # despite safe maximizers SMAC still sometimes suggests invalid
                    # configurations...
                    res: BenchQueryResult = self.objective(info.config.get_dictionary())
                    value = TrialValue(res.val_performance, time=0.5)

                    evaluations.append(res)
                    configs.append(info.config.get_dictionary())
                    self.hist.append((info.config.get_dictionary(), res, i))

                    self.smac.tell(info, value)
            except StopIteration:
                # if this exception is thrown, SMAC is not able to find better config.
                # add last config to fill until self._num_iter (for visualization purposes)
                last_entry = self.hist[-1]
                self.hist.append(last_entry)

        val_accs = [res.val_performance for res in evaluations]
        best = max(val_accs)
        best_idx = val_accs.index(best)
        return configs[best_idx], best
    
    @property
    def history(self):
        return self.hist