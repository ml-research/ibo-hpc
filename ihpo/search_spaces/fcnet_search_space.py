from .search_space import SearchSpace
import numpy as np
from spn.structure.StatisticalTypes import MetaType
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, CategoricalHyperparameter
from syne_tune.config_space import Categorical, Integer
from skopt.space import Categorical as CategoricalSkopt
from copy import deepcopy

class FCNetSearchSpace(SearchSpace):

    def __init__(self) -> None:
        super().__init__()
        self.search_space_definition = self.get_search_space_definition()

    def sample(self, size=1, keep_domain=True, exclude_fidelities=False, **kwargs):
        samples = []
        for _ in range(size):
            x_sample = {}
            for name, val in self.search_space_definition.items():
                if val['dtype'] == 'int':
                    if val['is_log']:
                        x = np.random.uniform(min(val['allowed']), max(val['allowed']))[0]
                    else:
                        x = np.exp(np.random.uniform(min(val['allowed']), max(val['allowed'])))[0]
                else:
                    x = np.random.choice(val['allowed'], size=1)[0]
                x_sample[name] = x
            
            # convert numpy-datatypes to Python-native datatypes. Required because FCNet is tabular and needs hashable types.
            x_sample = {key: value.item() if isinstance(value, np.generic) else value.tolist() if isinstance(value, np.ndarray) else value
                  for key, value in x_sample.items()}
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
            if val['dtype'] == 'int':
                hp = UniformIntegerHyperparameter(key, min(val['allowed']), max(val['allowed']), default_value=min(val['allowed']))
            else:
                hp = CategoricalHyperparameter(key, val['allowed'])
            config_space.add_hyperparameter(hp)
        return config_space
    
    def to_synetune(self):
        config_space = {}
        for key, val in self.config_space.items():
            if val['dtype'] == 'int':
                hp = Integer(min(val['allowed']), max(val['allowed']))
            else:
                hp = Categorical(val['allowed'])
            config_space[key] = hp
                    
        return config_space
    
    def to_hypermapper(self):
        config_space = {}
        for key, val in self.search_space_definition.items():
            if val['dtype'] == 'int':
                hp = {
                        "parameter_type": "ordinal",
                        "values": [min(val['allowed']), max(val['allowed'])]
                    }
            else:
                hp = {
                        "parameter_type": "categorical",
                        "values": val['allowed']
                    }
            config_space[key] = hp
        return config_space
    
    def is_valid(self, config):
        return True
    
    def get_search_space_definition(self):
        search_space_definition = {
            'activation_fn_1': {
                'type': MetaType.DISCRETE,
                'allowed': ['relu', 'tanh'],
                'dtype': 'str',
                'is_log': False,
            },
            'activation_fn_2': {
                'type': MetaType.DISCRETE,
                'allowed': ['relu', 'tanh'],
                'dtype': 'str',
                'is_log': False,
            },
            'batch_size': {
                'type': MetaType.DISCRETE,
                'allowed': [8, 16, 32, 64],
                'dtype': 'cat',
                'is_log': False,
            },
            'dropout_1': {
                'type': MetaType.DISCRETE,
                'allowed': [0.0, 0.3, 0.6],
                'dtype': 'cat',
                'is_log': False,
            },
            'dropout_2': {
                'type': MetaType.DISCRETE,
                'allowed': [0.0, 0.3, 0.6],
                'dtype': 'cat',
                'is_log': False,
            },
            'init_lr' : {
                'type': MetaType.DISCRETE,
                'allowed': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                'dtype': 'cat',
                'is_log': False,
            },
            'lr_schedule': {
                'type': MetaType.DISCRETE,
                'allowed': ['cosine', 'fix'],
                'dtype': 'str',
                'is_log': False,
            },
            'n_units_1': {
                'type': MetaType.DISCRETE,
                'allowed': [16, 32, 64, 128, 256, 512],
                'dtype': 'cat',
                'is_log': False,
            },
            'n_units_2': {
                'type': MetaType.DISCRETE,
                'allowed': [16, 32, 64, 128, 256, 512],
                'dtype': 'cat',
                'is_log': False,
            }
        }
        return search_space_definition

    def change_search_space(self, key, value):
        self.search_space_definition[key] = value
    
