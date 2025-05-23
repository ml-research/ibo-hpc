from typing import Dict
from ..search_spaces import SearchSpace

class Optimizer:

    def __init__(self, search_space, objective, seed=0) -> None:
        self.search_space = search_space
        self.objective = objective
        self.evaluations = []
        self.seed = seed

    def optimize(self):
        raise NotImplementedError('Optimizer base class has no optimize implementation')
    
    def intervene(self, intervention: Dict):
        raise NotImplementedError('Optimizer base class has no intervene implementation')
    
    def adapt_to_search_space(self, new_search_space: Dict):
        raise NotImplementedError('Optimizer base class has no adapt_to_search_space implementation')
    
    def set_search_space(self, search_space: SearchSpace):
        self.search_space = search_space
        self.evaluations = [] 
    
    @property
    def history(self):
        raise NotImplementedError('Optimizer base class does not implement history')