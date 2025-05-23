from .search_space import SearchSpace
from jahs_bench.api import Benchmark
import numpy as np
from spn.structure.StatisticalTypes import MetaType
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from syne_tune.config_space import Float

class BraninSearchSpace(SearchSpace):

    def __init__(self) -> None:
        super().__init__()
        self.config_space = {
            'x_1': [-5, 10],
            'x_2': [0, 15],
        }
        self.search_space_definition = self.get_search_space_definition()

    def sample(self, size=1, keep_domain=True, exclude_fidelities=False, **kwargs):
        samples = []
        for _ in range(size):
            x1 = np.random.uniform(self.config_space['x_1'][0], self.config_space['x_1'][1])
            x2 = np.random.uniform(self.config_space['x_2'][0], self.config_space['x_2'][1])
            samples.append({'x_1': x1, 'x_2': x2})
        return samples
    
    def get_neighbors(self, config: dict, num_cont_neighbors=6):
        raise NotImplementedError('Not implemented')
    
    def to_configspace(self):
        exclude_from_config_space = ['Optimizer'] #['Optimizer', 'W', 'N', 'Resolution']
        config_space = ConfigurationSpace()
        for key, val in self.config_space.items():
            if key not in exclude_from_config_space:
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
        # IMPORTANT: DO NOT CHANGE ORDER! MUST BE PROVIDED IN EXACTLY THIS ORDER
        # TO THE SPFLOW CONTEXT!
        search_space_definition = {
            'x_1': {
                'type': MetaType.REAL,
                'dtype': 'float',
                'min': self.config_space['x_1'][0],
                'max':self.config_space['x_1'][1],
            },
            'x_2': {
                'type': MetaType.REAL,
                'dtype': 'float',
                'min': self.config_space['x_2'][0],
                'max': self.config_space['x_2'][1]
            }
        }
        return search_space_definition

    def change_search_space(self, key, value):
        self.config_space[key] = value
    
