from .nas_search_space import NASLibSearchSpace
from naslib.search_spaces import NasBench101SearchSpace
import numpy as np
from spn.structure.StatisticalTypes import MetaType
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
import numpy as np
from copy import deepcopy
from syne_tune.config_space import Categorical
from skopt.space import Categorical as CategoricalSkopt

class NAS101SearchSpace(NASLibSearchSpace):

    def __init__(self, benchmark: NasBench101SearchSpace,
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
        return samples


    def get_search_space_definition(self):
        search_space_repr = {}
        num_vertices = 7
        li, ri = np.triu_indices(num_vertices, 1)
        triu_ind = np.array([[i, j] for i, j in zip(li, ri)])
        for i, j in triu_ind:
            search_space_repr[f'e_{i}_{j}'] = {
                'type': MetaType.BINARY,
                'allowed': [0, 1],
                'dtype': 'int'
            }
        # add parameters for operation choice
        for i in range(1, num_vertices - 1):
            search_space_repr[f'o_{i}'] = {
                'type': MetaType.DISCRETE,
                'allowed': self.operations,
                'dtype': 'str'
            }
    
        return search_space_repr
    
    def to_configspace(self):
        config_space = ConfigurationSpace()
        num_vertices = 7
        li, ri = np.triu_indices(num_vertices, 1)
        triu_ind = np.array([[i, j] for i, j in zip(li, ri)])
        for i, j in triu_ind:
            hp = CategoricalHyperparameter(f'e_{i}_{j}', [0, 1])
            config_space.add_hyperparameter(hp)
        for i in range(num_vertices - 2): # num_vertices - 2 since input and output are fixed
            hp = CategoricalHyperparameter(f'o_{i}', self.operations)
            config_space.add_hyperparameter(hp)
    
        return config_space
    
    def to_synetune(self):
        config_space = {}
        num_vertices = 7
        li, ri = np.triu_indices(num_vertices, 1)
        triu_ind = np.array([[i, j] for i, j in zip(li, ri)])
        for i, j in triu_ind:
            hp = Categorical([0, 1])
            config_space[f'e_{i}_{j}'] = hp
        for i in range(num_vertices - 2): # num_vertices - 2 since input and output are fixed
            hp = Categorical(self.operations)
            config_space[f'o_{i}'] = hp
    
        return config_space
    
    def to_hypermapper(self):
        config_space = {}
        num_vertices = 7
        li, ri = np.triu_indices(num_vertices, 1)
        triu_ind = np.array([[i, j] for i, j in zip(li, ri)])
        for i, j in triu_ind:
            hp = {
                "parameter_type": "ordinal",
                "values": [0, 1]
            }
            config_space[f'e_{i}_{j}'] = hp
        for i in range(num_vertices - 2): # num_vertices - 2 since input and output are fixed
            hp = {
                "parameter_type": "categorical",
                "values": self.operations
            }
            config_space[f'o_{i}'] = hp
    
        return config_space
    
    def to_skopt(self):
        hp_def = self.to_configspace()
        space = []
        for hp_name in hp_def.get_all_unconditional_hyperparameters():
            hp = hp_def.get_hyperparameter(hp_name)
            skopt_hp = CategoricalSkopt(hp.choices, name=hp.name)
            space.append(skopt_hp)
        return space
    
    def to_dict(self, benchmark: NasBench101SearchSpace):
        arch_descr = benchmark.get_spec()
        matrix, ops = arch_descr['matrix'], arch_descr['ops']
        sample = self._nas101_adj_to_dict(matrix, ops)
        return sample
    
    def is_valid(self, cfg):
        spec_dict = self._create_spec(cfg)
        spec = self.dataset_api['api'].ModelSpec(matrix=spec_dict['matrix'], ops=spec_dict['ops'])
        return self.dataset_api['nb101_data'].is_valid(spec)
    
    def _create_spec(self, cfg):
        adj = np.zeros((7, 7))
        ops = []
        for key, val in cfg.items():
            if key.startswith('e_'):
                i, j = key.split('_')[1:]
                adj[int(i), int(j)] = val
            elif key.startswith('o_'):
                i = key.split('_')[-1]
                ops.append(val)
        adj = adj.astype(np.uint8)
        operations = ['input'] + ops + ['output']
        return {'matrix': adj, 'ops': operations}

    def _nas101_adj_to_dict(self, adj, ops):
        dictionary = {}
        li, ri = np.triu_indices(7, 1)
        triu_ind = np.array([[i, j] for i, j in zip(li, ri)])
        for i,j in triu_ind:
            dictionary[f'e_{i}_{j}'] = adj[i, j]
        
        for i in range(1, adj.shape[0] - 1):
            dictionary[f'o_{i}'] = ops[i]

        return dictionary
    
    @property
    def operations(self):
        return ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
