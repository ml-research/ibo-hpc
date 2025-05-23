from .search_space import SearchSpace
import numpy as np
from spn.structure.StatisticalTypes import MetaType
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from syne_tune.config_space import Float, loguniform, Integer
from copy import deepcopy

class LCSearchSpace(SearchSpace):

    def __init__(self) -> None:
        super().__init__()
        self.search_space_definition = self.get_search_space_definition()

    def sample(self, size=1, keep_domain=True, exclude_fidelities=False, **kwargs):
        samples = []
        for _ in range(size):
            x_sample = {}
            for name, val in self.search_space_definition.items():
                if val['dtype'] == 'float':
                    if not val['is_log']:
                        x = np.random.uniform(val['min'], val['max'])
                    else:
                        x = np.exp(np.random.uniform(np.log(val['min']), np.log(val['max'])))
                elif val['dtype'] == 'int':
                    x = np.random.randint(min(val['allowed']), max(val['allowed']))
                x_sample[name] = x
                
            samples.append(x_sample)
        return samples
    
    def get_neighbors(self, config: dict, num_cont_neighbors=5):
        neighbors = []
        if config is None:
            return self.sample()
        for name, param_def in self.search_space_definition.items():
            curr_value = config[name]
            # handle continuous parameters
            if param_def['dtype'] == 'float':
                min_val = self.search_space_definition[name]['min']
                max_val = self.search_space_definition[name]['max']
                dist = max_val - min_val
                rnd_left = np.random.uniform(0, 0.2*dist, size=(num_cont_neighbors // 2))
                rnd_right = np.random.uniform(0, -0.2*dist, size=(num_cont_neighbors // 2))
                rnd = np.concatenate((rnd_left, rnd_right))
                rnd += curr_value
                ns = list(rnd)
                for neighbor in ns:
                    curr_config = deepcopy(config)
                    curr_config[name] = neighbor
                    neighbors.append(curr_config)
            else:
                options = self.search_space_definition[name]['allowed']
                num_options = len(options)
                ns = []
                curr_idx = options.index(curr_value)
                if num_options > 2:
                    lidx, ridx = (curr_idx - 1) % num_options, (curr_idx + 1) % num_options
                    ns.append(options[lidx])
                    ns.append(options[ridx])
                else:
                    ns.append(options[(curr_idx + 1) % 2])
                
                for neighbor in ns:
                    curr_config = deepcopy(config)
                    curr_config[name] = neighbor
                    neighbors.append(curr_config)
        return neighbors
    
    def to_configspace(self):
        config_space = ConfigurationSpace()
        for key, val in self.search_space_definition.items():
            if val['dtype'] == 'float':
                hp = UniformFloatHyperparameter(key, val['min'], val['max'], log=val['is_log'])
            elif val['dtype'] == 'int':
                hp = UniformIntegerHyperparameter(key, min(val['allowed']), max(val['allowed']), default_value=val['allowed'][0])
            config_space.add_hyperparameter(hp)
        return config_space
    
    def to_synetune(self):
        config_space = {}
        for key, val in self.config_space.items():
            if val['dtype'] == 'float':
                if val['is_log']:
                    hp = loguniform(val['min'], val['max'])
                else:
                    hp = Float(val['min'], val['max'])
            elif val['dtype'] == 'int':
                hp = Integer(min(val['allowed']), max(val['allowed']))
            config_space[key] = hp
                    
        return config_space
    
    def to_hypermapper(self):
        config_space = {}
        for key, val in self.search_space_definition.items():
            if val['dtype'] == 'float':
                hp = {
                        "parameter_type": "real",
                        "values": [val['min'], val['max']]
                    }
            elif val['dtype'] == 'int':
                    hp = {
                        "parameter_type": "ordinal",
                        "values": val['allowed']
                    }
            config_space[key] = hp
        return config_space
    
    def is_valid(self, config):
        return True
    
    def get_search_space_definition(self):
        search_space_definition = {
            'num_layers': {
                'type': MetaType.DISCRETE,
                'allowed': [1, 2, 3, 4, 5],
                'dtype': 'int',
                'is_log': False,
            }, 
            'max_units': {
                'type': MetaType.DISCRETE,
                'allowed': list(range(64, 1024)), # in paper it's [64, 512], but in data the upper bound is 1024
                'dtype': 'int',
                'is_log': False,
            }, 
            'batch_size': {
                'type': MetaType.DISCRETE,
                'allowed': list(range(16, 512)),
                'dtype': 'int',
                'is_log': False,
            }, 
            'learning_rate': {
                'type': MetaType.REAL,
                'min': 1e-4,
                'max': 0.1,
                'dtype': 'float',
                'is_log': False,
            }, 
            'momentum': {
                'type': MetaType.REAL,
                'min': 0.1,
                'max': 0.99,
                'dtype': 'float',
                'is_log': False,
            }, 
            'max_dropout': {
                'type': MetaType.REAL,
                'min': 0.0,
                'max': 1.0,
                'dtype': 'float',
                'is_log': False,
            }, 
            'weight_decay': {
                'type': MetaType.REAL,
                'min': 1e-5,
                'max': 0.1,
                'dtype': 'float',
                'is_log': False,
            }
        }
        return search_space_definition

    def change_search_space(self, key, value):
        self.search_space_definition[key] = value
    
