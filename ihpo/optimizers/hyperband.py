from ..optimizers import Optimizer
from smac import Scenario, HyperbandFacade
from ..search_spaces import SearchSpace
from ..benchmarks import BenchQueryResult
from ConfigSpace import Configuration
from ..utils.smac import HyperbandConfigSelector
from smac.runhistory.dataclasses import TrialValue

class HyperbandOptimizer(Optimizer):

    def __init__(self, search_space: SearchSpace, objective, 
                 **scenario_args) -> None:
        super().__init__(search_space, objective)
        self._num_iter = scenario_args['n_trials']
        self.scenario = Scenario(search_space.to_configspace(), **scenario_args)
        self.objective = objective
        config_selector = HyperbandConfigSelector(
            self.scenario, 
            search_space
        )
        intensifier = HyperbandFacade.get_intensifier(
            self.scenario, 
            eta=2
        )
        self.smac = HyperbandFacade(
            self.scenario,
            self._objective_wrapper,
            config_selector=config_selector,
            overwrite=True,
            intensifier=intensifier
        )
        self.hist = []
        self.search_space = search_space

    def _objective_wrapper(self, config: Configuration,
                           budget,
                           seed: int = 0) -> float:
        res: BenchQueryResult = self.objective(config.get_dictionary(), budget=budget)
        return res.val_performance

    def optimize(self):
        """
            Perform Hyperband Optimization.
        """
        evaluations, configs = [], []
        for i in range(self._num_iter):
            if i % 50 == 0:
                print(f"Iteration {i}/{self._num_iter}")
            try:
                info = self.smac.ask()

                if self.search_space.is_valid(info.config):
                    # only evaluate valid configs
                    # despite safe maximizers Hyperband still sometimes suggests invalid
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