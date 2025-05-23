from .optimizer import Optimizer
from smac import Scenario, HyperparameterOptimizationFacade
from ..search_spaces import SearchSpace
from ..benchmarks import BenchQueryResult
from ConfigSpace import Configuration
from smac.runhistory.dataclasses import TrialValue, TrialInfo
from smac.main.config_selector import ConfigSelector
from smac.model.random_forest.random_forest import RandomForest
from ..utils.smac import SafeLocalAndSrotedRandomSearch
from time import time

class SMACOptimizer(Optimizer):

    def __init__(self, search_space: SearchSpace, objective, 
                 cfg_selector_retries=8, log_hpo_runtime=False, **scenario_args) -> None:
        super().__init__(search_space, objective)
        self._num_iter = scenario_args['n_trials']
        self._log_hpo_runtime = log_hpo_runtime
        self.hpo_runtimes = []
        self.scenario = Scenario(search_space.to_configspace(), **scenario_args)
        self.objective = objective
        model = RandomForest(
            search_space.to_configspace(),
            n_trees=10,
        )
        maximizer = SafeLocalAndSrotedRandomSearch(
            search_space,
            challengers=10000,
            local_search_iterations=20,
            n_steps_plateau_walk=20,
            seed=scenario_args['seed']
        )
        initial_design = HyperparameterOptimizationFacade.get_initial_design(self.scenario, n_configs=20)
        intensifier = HyperparameterOptimizationFacade.get_intensifier(
            self.scenario,
            max_config_calls=1,  # We basically use one seed per config only
        )
        self.smac = HyperparameterOptimizationFacade(
            self.scenario,
            self._objective_wrapper,
            acquisition_maximizer=maximizer,
            overwrite=True,
            model=model,
            config_selector=ConfigSelector(
                self.scenario, retries=cfg_selector_retries, retrain_after=2
            ),
            intensifier=intensifier,
            initial_design=initial_design
        )
        self.search_space = search_space
        self.hist = []

    def _objective_wrapper(self, config: Configuration, 
                           seed) -> float:
        res: BenchQueryResult = self.objective(config)
        return -res.val_performance

    def optimize(self):
        """
            Perform one step of SMAC.
        """
        evaluations, configs = [], []
        for i in range(self._num_iter):
            if i % 50 == 0:
                print(f"Iteration {i}/{self._num_iter}")
            try:
                start_time = time() if self._log_hpo_runtime else None
                info = self.smac.ask()
                
                if self._log_hpo_runtime:
                    end_time = time()
                    diff_ask = end_time - start_time

                if self.search_space.is_valid(info.config):
                    # only evaluate valid configs
                    # despite safe maximizers SMAC still sometimes suggests invalid
                    # configurations...
                    res: BenchQueryResult = self.objective(info.config.get_dictionary())
                    value = TrialValue(cost=-res.val_performance, time=0.5)

                    evaluations.append(res)
                    configs.append(info.config.get_dictionary())
                    self.hist.append((info.config.get_dictionary(), res, i))

                    start_time = time() if self._log_hpo_runtime else None
                    self.smac.tell(info, value)

                    if self._log_hpo_runtime:
                        end_time = time()
                        diff_tell = end_time - start_time
                        diff = diff_ask + diff_tell
                        self.hpo_runtimes.append(diff)
            except StopIteration:
                # add random configuration if SMAC fails
                cfg = self.search_space.to_configspace().sample_configuration()
                cfg_dict = cfg.get_dictionary()
                res = self.objective(cfg_dict)
                evaluations.append(res)
                configs.append(cfg_dict)
                self.hist.append((cfg_dict, res, i))
                value = TrialValue(-res.val_performance, time=0.5)
                info = TrialInfo(cfg)
                self.smac.tell(info, value)

        val_accs = [res.val_performance for res in evaluations]
        best = max(val_accs)
        best_idx = val_accs.index(best)
        return configs[best_idx], best
    
    @property
    def history(self):
        if self._log_hpo_runtime:
            evaluations = []
            hpo_runtime = sum(self.hpo_runtimes)
            for cfg, eval_res, it in self.hist:
                eval_res.set_optim_cost(hpo_runtime)
                evaluations.append((cfg, eval_res, it))
            return evaluations
        return self.hist