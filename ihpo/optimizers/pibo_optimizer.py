from typing import Dict
from ..optimizers import Optimizer
from smac import Scenario, HyperparameterOptimizationFacade
from smac.acquisition.function import PriorAcquisitionFunction, EI
from ConfigSpace import Configuration, ConfigurationSpace
from ..benchmarks import BenchQueryResult
from ..search_spaces import SearchSpace
from ..utils.smac import SafeLocalAndSrotedRandomSearch
from smac.main.config_selector import ConfigSelector
from smac.runhistory.dataclasses import TrialValue, TrialInfo

class PiBOptimizer(Optimizer):

    def __init__(self, search_space: SearchSpace, objective, **scenario_params) -> None:
        super().__init__(search_space, objective)
        self.config_space = search_space.to_configspace()
        self.scenario = Scenario(self.config_space, **scenario_params)
        self.objective = objective
        self._scenario_args = scenario_params
        self._num_iter = scenario_params['n_trials']
        self.search_space = search_space
        self.hist = []
        self._init_smac_objects()

    def _init_smac_objects(self):
        maximizer = SafeLocalAndSrotedRandomSearch(
            self.search_space,
            challengers=10000,
            local_search_iterations=20,
            n_steps_plateau_walk=20,
            seed=self._scenario_args['seed']
        )
        intensifier = HyperparameterOptimizationFacade.get_intensifier(
            self.scenario,
            max_config_calls=3,
        )
        self.acquisition = PriorAcquisitionFunction(
            HyperparameterOptimizationFacade.get_acquisition_function(self.scenario),
            self._num_iter / 10
        )
        cfg_selector_retries=16
        self.smac = HyperparameterOptimizationFacade(
            self.scenario,
            self._objective_wrapper,
            intensifier=intensifier,
            acquisition_function=self.acquisition,
            acquisition_maximizer=maximizer,
            overwrite=True,
            config_selector=ConfigSelector(
                self.scenario, retries=cfg_selector_retries
            )
        )

    def _objective_wrapper(self, config: Configuration, 
                           seed: int = 0) -> float:
        res: BenchQueryResult = self.objective(config)
        return -res.val_performance
    
    def optimize(self):
        """
            Perform one step of PiBO.
        """
        evaluations, configs = [], []
        for i in range(self._num_iter):
            if i % 50 == 0:
                print(f"Iteration {i}/{self._num_iter}")
                #print(self.history)
            try:
                info = self.smac.ask()

                if self.search_space.is_valid(info.config):
                    # only evaluate valid configs
                    # despite safe maximizers SMAC still sometimes suggests invalid
                    # configurations...
                    res: BenchQueryResult = self.objective(info.config.get_dictionary())
                    value = TrialValue(-res.val_performance, time=0.5)

                    evaluations.append(res)
                    configs.append(info.config.get_dictionary())
                    self.hist.append((info.config.get_dictionary(), res, i))

                    self.smac.tell(info, value)
            except StopIteration:
                # add random configuration if PiBO fails
                cfg = self.search_space.to_configspace().sample_configuration()
                res = self.objective(cfg.get_dictionary())
                evaluations.append(res)
                configs.append(cfg.get_dictionary())
                self.hist.append((cfg.get_dictionary(), res, i))
                value = TrialValue(-res.val_performance, time=0.5)
                info = TrialInfo(cfg)
                self.smac.tell(info, value)
                #last_entry = self.hist[-1]
                #self.hist.append(last_entry)

        val_accs = [res.val_performance for res in evaluations]
        best = max(val_accs)
        best_idx = val_accs.index(best)
        return configs[best_idx], best
    
    @property
    def history(self):
        return self.hist
    
    def intervene(self, intervention: Dict, **kwargs):
        """
            NOTE: **kwargs only due to compatbility to other optimizers
        """
        new_hyperparams = []
        for hp in self.config_space.get_hyperparameters():
            if hp.name in intervention:
                new_hyperparams.append(intervention[hp.name])
            else:
                new_hyperparams.append(hp)
        
        cs = ConfigurationSpace()
        cs.add_hyperparameters(new_hyperparams)
        self.config_space = cs
        self.scenario = Scenario(cs, **self._scenario_args)
        self._init_smac_objects()