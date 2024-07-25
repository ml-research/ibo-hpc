from .nas_search_space import NASLibSearchSpace
from naslib.search_spaces import TransBench101SearchSpaceMicro
from typing import Union
import numpy as np
from spn.structure.StatisticalTypes import MetaType
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
import numpy as np
from copy import deepcopy
from syne_tune.config_space import Categorical

class TransNASSearchSpace(NASLibSearchSpace):

    def __init__(self, benchmark:  TransBench101SearchSpaceMicro,
                dataset_api) -> None:
        super().__init__(benchmark, dataset_api)
        # in case of NASLib the benchmark and search space are the same thing
        # However, for compatibility it's better to have a spearate SearchSpace
        # class
        self.benchmark = benchmark

        self.dataset_api = dataset_api
        self.search_space_definition = self.get_search_space_definition()

    def sample(self, size=1, keep_domain=True, **kwargs):
        samples = []
        for _ in range(size):
            new_bench = self.benchmark.clone()
            new_bench.sample_random_architecture(dataset_api=self.dataset_api)
            sample = self.to_dict(new_bench)
            samples.append(sample)
        #if keep_domain:
        #    return samples
        #processed_configs = []
        #for sample in samples:
        #    processed_config = deepcopy(sample)
        #    for name, val in sample.items():
        #        if name.startswith('Op'):
        #            idx = self.search_space_definition[name]['allowed'].index(val)
        #            processed_config[name] = idx
        #    processed_configs.append(processed_config)
        #return processed_configs
        return samples

    def get_search_space_definition(self):
        search_space_repr = {}
        num_edges = 6
        for i in range(num_edges):
            search_space_repr[f'Op_{i}'] = {
                'type': MetaType.DISCRETE,
                'allowed': self.operations
            }
    
        return search_space_repr
    
    def to_configspace(self):
        config_space = ConfigurationSpace()
        num_edges = 6
        for i in range(num_edges):
            hp = CategoricalHyperparameter(f'Op_{i}', self.operations)
            config_space.add_hyperparameter(hp)
    
        return config_space
    
    def to_synetune(self):
        config_space = {}
        num_edges = 6
        for i in range(num_edges):
            hp = Categorical(self.operations)
            config_space[f'Op_{i}'] = hp
    
        return config_space
    
    def to_dict(self, benchmark: TransBench101SearchSpaceMicro):
        indices = benchmark.get_op_indices()
        sample = {f'Op_{i}': o for i, o in enumerate(indices)}
        return sample
    
    def is_valid(self, cfg):
        return True
    
    def _create_spec(self, cfg):
        return list(cfg.values())

    @property
    def operations(self):
        return list(range(4))