from .search_space import SearchSpace
from jahs_bench.api import Benchmark
from copy import deepcopy
import numpy as np
from spn.structure.StatisticalTypes import MetaType
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from syne_tune.config_space import Float, Categorical
from skopt.space import Real as RealSkopt
from skopt.space import Integer as IntegerSkopt 
from skopt.space import Categorical as CategoricalSkopt

class JAHSBenchSearchSpace(SearchSpace):

    def __init__(self, benchmark: Benchmark) -> None:
        super().__init__()
        self.config_space = {
            #'Optimizer': 'SGD',
            'LearningRate': [1e-3, 1e0],
            'WeightDecay': [1e-5, 1e-2],
            #'LearningRate': [1e-6, 1e1],
            #'WeightDecay': [1e-7, 1e0],
            'Activation': ['Mish', 'ReLU', 'Hardswish'],
            'TrivialAugment': [True, False],
            'Op1': list(range(5)),
            'Op2': list(range(5)),
            'Op3': list(range(5)),
            'Op4': list(range(5)),
            'Op5': list(range(5)),
            'Op6': list(range(5)),
            'N': list(range(16)), #S1 
            #'N': list(range(8)), #S2 
            #'N': [1, 3, 5], #S3
            'W': list(range(32)), #S1 
            #'W': list(range(3, 17)), #S2 
            #'W': [4, 8, 16], #S3 
            'epoch': list(range(200)),
            'Resolution': [0, 1], #S1
            #'Resolution': [0.15, 0.17, 0.19, 0.2, 0.22, 0.24, 0.25, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.9, 0.92, 0.94, 0.96, 0.98, 1.], #S2
            #'Resolution': [0.25, 0.5, 1.] #S3
        }
        self._exclude_from_config_space = ['Optimizer']
        self.benchmark = benchmark
        self.search_space_definition = self.get_search_space_definition()

    def sample(self, size=1, keep_domain=True, exclude_fidelities=False, **kwargs):
        samples = []
        if exclude_fidelities:
            for _ in range(size):
                cfg = self.benchmark.sample_config()
                cfg.pop('N', None)
                cfg.pop('W', None)
                cfg.pop('Resolution', None)
                cfg.pop('Optimizer', None)
                cfg.pop('epoch', None)
                samples.append(cfg)
        else:
            config_space = self.to_configspace()
            samples = [config_space.sample_configuration().get_dictionary() for _ in range(size)]
            #samples = [self.benchmark.sample_config() for _ in range(size)]
        
        return samples
    
    def get_neighbors(self, config: dict, num_cont_neighbors=6):
        if config is None:
            rand_configs = self.sample(num_cont_neighbors)
            return rand_configs
        # we do not generate neighbors for optimizers
        exclude_from_generation = ['Optimizer'] #['Optimizer', 'W', 'N', 'Resolution']
        searchable_params = [k for k in self.config_space.keys() if k not in exclude_from_generation]
        neighbors = []
        for name in searchable_params:
            curr_value = config[name]
            # handle continuous parameters
            if name in ['LearningRate', 'WeightDecay', 'Resolution']:
                min_val = self.config_space[name][0]
                max_val = self.config_space[name][1]
                dist = max_val - min_val
                rnd_left = np.random.uniform(0, 0.2*dist, size=(num_cont_neighbors // 2))
                rnd_right = np.random.uniform(0, -0.2*dist, size=(num_cont_neighbors // 2))
                rnd = np.concatenate((rnd_left, rnd_right))
                rnd += curr_value
                ns = list(rnd)
                for neighbor in ns:
                    curr_config = deepcopy(config)
                    if name == 'Resolution':
                        # bound between 0 and 1
                        if neighbor < 0:
                            neighbor = 0
                        elif neighbor > 1:
                            neighbor = 1
                    curr_config[name] = neighbor
                    neighbors.append(curr_config)
            else:
                num_options = len(self.config_space[name])
                options = self.config_space[name]
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
        exclude_from_config_space = ['Optimizer'] #['Optimizer', 'W', 'N', 'Resolution']
        config_space = ConfigurationSpace()
        for key, val in self.config_space.items():
            if key not in exclude_from_config_space:
                if key in ['LearningRate', 'WeightDecay', 'Resolution']:
                #if key in ['LearningRate', 'WeightDecay']:
                    hp = UniformFloatHyperparameter(key, val[0], val[1])
                elif key == 'epoch':
                    hp = UniformIntegerHyperparameter(key, val[0], val[-1], default_value=val[-1])
                else:
                    hp = CategoricalHyperparameter(key, val)

                config_space.add_hyperparameter(hp)
        
        return config_space
    
    def to_synetune(self):
        config_space = {}
        for key, val in self.config_space.items():
            if key not in self._exclude_from_config_space:
                if key in ['LearningRate', 'WeightDecay', 'Resolution']:
                    hp = Float(val[0], val[1])
                    config_space[key] = hp
                else:
                    hp = Categorical(val)
                    config_space[key] = hp
        return config_space
    
    def to_hypermapper(self):
        config_space = {}
        for key, val in self.config_space.items():
            if key not in self._exclude_from_config_space:
                if key in ['LearningRate', 'WeightDecay', 'Resolution']:
                    hp = {
                        "parameter_type": "real",
                        "values": val
                    }
                elif key in ['W', 'N', 'epoch'] or key.startswith('O'):
                    hp = {
                        "parameter_type": "ordinal",
                        "values": val
                    }
                elif key == 'TrivialAugment':
                    hp = {
                        "parameter_type": "ordinal",
                        "values": [1, 0]
                    }
                else:
                    hp = {
                        "parameter_type": "categorical",
                        "values": val
                    }
            config_space[key] = hp
        return config_space
    
    def to_skopt(self, all_numeric=True):
        hp_def = self.to_configspace()
        space = []
        for hp_name in hp_def.get_all_unconditional_hyperparameters():
            hp = hp_def.get_hyperparameter(hp_name)
            if isinstance(hp, CategoricalHyperparameter):
                choices = list(range(len(hp.choices))) if all_numeric else hp.choices
                skopt_hp = CategoricalSkopt(choices, name=hp.name)
                space.append(skopt_hp)
            elif isinstance(hp, UniformFloatHyperparameter):
                skopt_hp = RealSkopt(hp.lower, hp.upper, name=hp.name)
                space.append(skopt_hp)
            elif isinstance(hp, UniformIntegerHyperparameter):
                skopt_hp = IntegerSkopt(hp.lower, hp.upper, name=hp.name)
                space.append(skopt_hp)
        return space
    
    def is_valid(self, config):
        return True
    
    def get_search_space_definition(self):
        # IMPORTANT: DO NOT CHANGE ORDER! MUST BE PROVIDED IN EXACTLY THIS ORDER
        # TO THE SPFLOW CONTEXT!
        search_space_definition = {
            'Activation': {
                'type': MetaType.DISCRETE,
                'allowed': self.config_space['Activation'],
                'dtype': 'str',
                'is_log': False,
            },
            'LearningRate': {
                'type': MetaType.REAL,
                'min': 1e-3,
                'max': 1.0,
                'dtype': 'float',
                'is_log': False,
            },
            'N': {
                'type': MetaType.DISCRETE,
                'allowed': self.config_space['N'],
                'dtype': 'int',
                'is_log': False,
            },
            'Op1': {
                'type': MetaType.DISCRETE,
                'allowed': self.config_space['Op1'],
                'dtype': 'int',
                'is_log': False,
            },
            'Op2': {
                'type': MetaType.DISCRETE,
                'allowed': self.config_space['Op2'],
                'dtype': 'int',
                'is_log': False,
            },
            'Op3': {
                'type': MetaType.DISCRETE,
                'allowed': self.config_space['Op3'],
                'dtype': 'int',
                'is_log': False,
            },
            'Op4': {
                'type': MetaType.DISCRETE,
                'allowed': self.config_space['Op4'],
                'dtype': 'int',
                'is_log': False,
            },
            'Op5': {
                'type': MetaType.DISCRETE,
                'allowed': self.config_space['Op5'],
                'dtype': 'int',
                'is_log': False,
            },
            'Op6': {
                'type': MetaType.DISCRETE,
                'allowed': self.config_space['Op6'],
                'dtype': 'int',
                'is_log': False,
            },
            'Resolution': {
                'type': MetaType.REAL,
                'min': 0.0,
                'max': 1.0,
                'dtype': 'float',
                'is_log': False,
            },
            #'Resolution': {
            #    'type': MetaType.DISCRETE,
            #    'allowed': self.config_space['Resolution']
            #},
            'TrivialAugment': {
                'type': MetaType.BINARY,
                'allowed': self.config_space['TrivialAugment'],
                'dtype': 'bool'
            },
            'W': {
                'type': MetaType.DISCRETE,
                'allowed': self.config_space['W'],
                'dtype': 'int',
                'is_log': False,
            },
            'WeightDecay': {
                'type': MetaType.REAL,
                'min': 1e-5,
                'max': 1e-2,
                'dtype': 'float',
                'is_log': False,
            },
            'epoch': {
                'type': MetaType.DISCRETE,
                'allowed': self.config_space['epoch'],
                'dtype': 'int',
                'is_log': False,
            }
        }
        return search_space_definition

    def change_search_space(self, key, value):
        self.config_space[key] = value