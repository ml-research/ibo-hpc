from .search_space import SearchSpace
from jahs_bench.api import Benchmark
import numpy as np
from ..consts.dtypes import MetaType
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from syne_tune.config_space import Float

class RealSearchSpace(SearchSpace):

    def __init__(self, n_dims=2, min_=-5, max_=5) -> None:
        super().__init__()
        self.config_space = {}
        for n in range(n_dims):
            self.config_space[f'x_{n+1}'] = [min_, max_]
        self.search_space_definition = self.get_search_space_definition()

    def sample(self, size=1, keep_domain=True, exclude_fidelities=False, **kwargs):
        samples = []
        for _ in range(size):
            x_sample = {}
            for name, val in self.config_space.items():
                x = np.random.uniform(val[0], val[1])
                x_sample[name] = x
            samples.append(x_sample)
        return samples
    
    def get_neighbors(self, config: dict, num_cont_neighbors=6):
        raise NotImplementedError('Not implemented')
    
    def to_configspace(self):
        config_space = ConfigurationSpace()
        for key, val in self.config_space.items():
            hp = UniformFloatHyperparameter(key, val[0], val[1])
            config_space.add_hyperparameter(hp)
        return config_space
    
    def to_synetune(self):
        config_space = {}
        for key, val in self.config_space.items():
            hp = Float(val[0], val[1])
            config_space[key] = hp
                    
        return config_space
    
    def to_hypermapper(self):
        config_space = {}
        for key, val in self.config_space.items():
            hp = {
                    "parameter_type": "real",
                    "values": val
                }
            config_space[key] = hp
        return config_space
    
    def is_valid(self, config):
        return True
    
    def get_search_space_definition(self):
        search_space_definition = {}
        for name, val in self.config_space.items():
            search_space_definition[name] = {
                'type': MetaType.REAL,
                'dtype': 'float',
                'min': val[0],
                'max': val[1],
                'is_log': False
            }
        return search_space_definition

    def change_search_space(self, key, value):
        self.config_space[key] = value
    
