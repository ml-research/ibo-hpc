from .search_space import SearchSpace
import json
import os
import numpy as np
from spn.structure.StatisticalTypes import MetaType
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from syne_tune.config_space import Float, Categorical
from skopt.space import Real as RealSkopt
from skopt.space import Integer as IntegerSkopt 
from skopt.space import Categorical as CategoricalSkopt
from copy import deepcopy
from ..consts import NO_HYPERPARAMETERS


class HPOBSearchSpace(SearchSpace):

    def __init__(self, json_path, search_space_id) -> None:
        file = os.path.join(json_path, 'meta-dataset-descriptors.json')
        with open(file, 'r') as f:
            self.search_spaces = json.load(f)
        self.search_space_json = self.search_spaces[search_space_id]
        self.search_space_def = self._create_search_space_dict()
        super().__init__()

    def _create_search_space_dict(self):
        search_space_def = {}
        hpob_search_space = self.search_space_json['variables']
        for var in hpob_search_space.keys():
            if var == 'booster':
                continue
            variable_def = hpob_search_space[var]
            if '.' in var:
                splits = var.split('.')
                var = '_'.join(splits)
            if var in NO_HYPERPARAMETERS:
                var_collision = [v for v in NO_HYPERPARAMETERS if v == var][0]
                search_def_name = '_' + var_collision
            else:
                search_def_name = var
            if 'min' in variable_def.keys():
                min_, max_ = variable_def['min'], variable_def['max']
                #search_space_def[search_def_name] = {
                #    'type': MetaType.REAL,
                #    'dtype': 'float',
                #    'min': min_,
                #    'max': max_
                #}
                if min_ == 0 and max_ == 1:
                    search_space_def[search_def_name] = {
                        'type': MetaType.REAL,
                        'dtype': 'float',
                        'min': min_,
                        'max': max_,
                        'is_log': False,
                    }
                elif int(min_) == min_ and int(max_) == max_:
                    search_space_def[search_def_name] = {
                        'type': MetaType.DISCRETE,
                        'dtype': 'int',
                        'allowed': list(range(int(min_), int(max_))),
                        'is_log': False,
                    }
                else:
                    search_space_def[search_def_name] = {
                        'type': MetaType.REAL,
                        'dtype': 'float',
                        'min': min_,
                        'max': max_,
                        'is_log': False,
                    }
            elif 'categories' in variable_def.keys():
                search_space_def[search_def_name] = {
                    'type': MetaType.DISCRETE,
                    'dtype': 'str',
                    'allowed': variable_def['categories'],
                    'is_log': False,
                }
        return search_space_def

    def sample(self, size=1, keep_domain=True, **kwargs):
        """
            Get random sample from search space.
        """
        configurations = []
        for _ in range(size):
            cfg = {}
            for name, d in self.search_space_def.items():
                if d['type'] == MetaType.REAL:
                    min_, max_ = d['min'], d['max']
                    val = np.random.uniform(min_, max_, size=1).flatten()[0]
                    cfg[name] = val
                elif d['type'] == MetaType.DISCRETE:
                    val = np.random.choice(d['allowed'], size=1)[0]
                    if keep_domain:
                        cfg[name] = val
                    else:
                        idx = d['allowed'].index(val)
                        cfg[name] = int(idx)
            configurations.append(cfg)
        return configurations
    
    def get_neighbors(self, config, num_cont_neighbors=5):
        neighbors = []
        if config is None:
            return self.sample()
        for name, param_def in self.search_space_def.items():
            curr_value = config[name]
            # handle continuous parameters
            if param_def['dtype'] == 'float':
                min_val = self.search_space_def[name]['min']
                max_val = self.search_space_def[name]['max']
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
                options = self.search_space_def[name]['allowed']
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
        for key, val in self.search_space_def.items():
            if val['dtype'] == 'float':
                hp = UniformFloatHyperparameter(key, val['min'], val['max'])
            elif val['dtype'] == 'int':
                min_, max_ = min(val['allowed']), max(val['allowed'])
                hp = UniformIntegerHyperparameter(key, min_, max_, default_value=min_)
            else:
                hp = CategoricalHyperparameter(key, val['allowed'])

            config_space.add_hyperparameter(hp)
        
        return config_space

    def to_hypermapper(self):
        config_space = {}
        for key, val in self.search_space_def.items():
            if val['dtype'] == 'float':
                hp = {
                        "parameter_type": "real",
                        "values": [val['min'], val['max']]
                    }
            elif val['dtype'] == 'int':
                # NOTE: Modelling ints as integers instead of categoricals is important!
                #   Large categorical spaces lead to very slow LS!
                hp = {
                    "parameter_type": "integer",
                    "values": [min(val['allowed']), max(val['allowed'])]
                }
            else:
                hp = {
                        "parameter_type": "categorical",
                        "values": val['allowed']
                    }
            config_space[key] = hp
        return config_space
    
    def to_synetune(self):
        config_space = {}
        for key, val in self.search_space_def.items():
            if val['dtype'] == 'float':
                hp = Float(val['min'], val['max'])
                config_space[key] = hp
            else:
                hp = Categorical(val['allowed'])
                config_space[key] = hp
        return config_space
    
    def to_skopt(self):
        hp_def = self.to_configspace()
        space = []
        for hp_name in hp_def.get_all_unconditional_hyperparameters():
            hp = hp_def.get_hyperparameter(hp_name)
            if isinstance(hp, CategoricalHyperparameter):
                skopt_hp = CategoricalSkopt(hp.choices, name=hp.name)
                space.append(skopt_hp)
            elif isinstance(hp, UniformFloatHyperparameter):
                skopt_hp = RealSkopt(hp.lower, hp.upper, name=hp.name)
                space.append(skopt_hp)
            elif isinstance(hp, UniformIntegerHyperparameter):
                skopt_hp = IntegerSkopt(hp.lower, hp.upper, name=hp.name)
                space.append(skopt_hp)
        return space

    def get_search_space_definition(self):
        return self.search_space_def
    
    def is_valid(self, config):
        # just for compatibility
        return True