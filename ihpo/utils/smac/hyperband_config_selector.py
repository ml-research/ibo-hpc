from typing import Iterator
from ConfigSpace import Configuration
from smac.main.config_selector import ConfigSelector
from smac.scenario import Scenario
from ihpo.search_spaces import SearchSpace

class HyperbandConfigSelector(ConfigSelector):

    def __init__(self, 
                 scenario: Scenario,
                 search_space: SearchSpace,
                 *, 
                 retrain_after: int = 8, 
                 retries: int = 16, 
                 min_trials: int = 1) -> None:
        super().__init__(scenario, 
                         retrain_after=retrain_after, 
                         retries=retries, 
                         min_trials=min_trials)
        self.search_sapce = search_space
    
    def __iter__(self) -> Iterator[Configuration]:
        for config in super().__iter__():
            if self.search_sapce.is_valid(config.get_dictionary()):
                yield config