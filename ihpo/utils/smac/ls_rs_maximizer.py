from ConfigSpace import Configuration, ConfigurationSpace
from smac.acquisition.function import AbstractAcquisitionFunction
from smac.acquisition.maximizer import LocalAndSortedRandomSearch
from typing import Union
from ihpo.search_spaces import SearchSpace

class SafeLocalAndSrotedRandomSearch(LocalAndSortedRandomSearch):
    """
        A safe version of LocalAndSortedRandomSearch maximizer used by SMAC (default)
        in case we optimize ony NASBench101.
        Safe means that we check for invalid configurations suggested by SMAC.
        If this is not done, SMAC often fails to identify a valid configuration since
        in NASBench101 only a small fraction of possible configurations are valid.
        See https://arxiv.org/pdf/1902.09635.pdf for details.
    """

    def __init__(self, 
                 search_space: SearchSpace,
                 acquisition_function: Union[AbstractAcquisitionFunction, None] = None, 
                 challengers: int = 5000, 
                 max_steps: Union[int,  None] = None, 
                 n_steps_plateau_walk: int = 10, 
                 local_search_iterations: int = 10, 
                 seed: int = 0) -> None:
        super().__init__(search_space.to_configspace(), 
                         acquisition_function, 
                         challengers, 
                         max_steps, 
                         n_steps_plateau_walk, 
                         local_search_iterations, 
                         seed)
        self._search_space = search_space
    
    def _maximize(self, previous_configs: list[Configuration], 
                  n_points: int) -> list[tuple[float, Configuration]]:
        candidates = super()._maximize(previous_configs, n_points)
        valid_candidates = [(s, c) for (s, c) in candidates if self._search_space.is_valid(c)]
        return valid_candidates

