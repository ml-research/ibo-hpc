from .search_space import SearchSpace
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, OrdinalHyperparameter, FloatHyperparameter, \
    IntegerHyperparameter, OrdinalHyperparameter, CategoricalHyperparameter
import numpy as np
from copy import deepcopy
from spn.structure.StatisticalTypes import MetaType
from syne_tune.config_space import Float, Integer, Ordinal

class HPOBenchTabularSearchSpace(SearchSpace):
    """
        Search space wrapper of HPOBench Tabular search space.
    """

    def __init__(self, config_space: ConfigurationSpace) -> None:
        super().__init__()
        self.config_space = config_space
        self.search_space_definition = self.get_search_space_definition()

    def sample(self, size=1, keep_domain=True, **kwargs):
        """
            :param kee_domain: As the HPOTabularBenchmark doesn't support surrogates to be queried,
                we have to ensure to learn surrogate models only over a valid domain.
                Hence each valid value from the search space is mapped to an integer
                respecting the order (e.g. 0.1 -> 0, 0.2 -> 1, 0.3 -> 3).
            :param size: Number of samples
            Sample a configuration from the config space.
        """
        samples = self.config_space.sample_configuration(size)
        if not isinstance(samples, list):
            samples = [samples]
        if keep_domain:
            return samples
        
        processed_configs = []
        for config in samples:
            processed_config = {}
            for name, val in config.items():
                idx = self.search_space_definition[name]['allowed'].index(val)
                processed_config[name] = idx
            processed_configs.append(processed_config)
        return processed_configs
    
    def get_neighbors(self, config: dict, num_neighbors=6):
        """
            get neighbors of a given config. 
        """
        if config is None:
            return self.sample(size=num_neighbors)
        hyperparams = self.config_space.get_hyperparameters()
        neighbors = []
        for hp in hyperparams:
            name = hp.name
            curr_value = config[name]
            # initialize ns to avoid linting error
            ns = []
            # in HPOBench hyperparameters are either integer or float typed
            if isinstance(hp, UniformIntegerHyperparameter):
                ns_left = [(curr_value - i) for i in range(num_neighbors // 2)]
                ns_right = [(curr_value + i) for i in range(num_neighbors // 2)]
                ns = ns_left + ns_right
                ns = [val for val in ns if val >= hp.lower and val <= hp.upper]
            elif isinstance(hp, UniformFloatHyperparameter):
                # sample num_neighbors in 20% of unit ball around curr_value
                rnd_left = np.random.uniform(0, 0.2*hp.upper, size=(num_neighbors // 2))
                rnd_right = np.random.uniform(0, -0.2*hp.upper, size=(num_neighbors // 2))
                rnd = np.concatenate((rnd_left, rnd_right))
                rnd += curr_value
                ns = list(rnd)
            elif isinstance(hp, OrdinalHyperparameter):
                choices = list(hp.sequence)
                curr_idx = choices.index(curr_value)
                lidx, ridx = curr_idx - 1, curr_idx + 1
                ns_left = [choices[lidx % len(choices)]]
                ns_right = [choices[ridx % len(choices)]]
                ns = ns_left + ns_right

            for neighbor in ns:
                curr_config = deepcopy(config)
                curr_config[name] = neighbor
                neighbors.append(curr_config)
        return neighbors

    def get_search_space_definition(self):
        hyperparams = self.config_space.get_hyperparameters_dict()
        search_space_repr = {}
        for name, hp in hyperparams.items():
            search_space_repr[name] = {
                'type': MetaType.DISCRETE,
                'allowed': list(hp.sequence)
            }
        return search_space_repr
    
    def to_configspace(self):
        """
            Convert search space to ConfigSpace object
        """
        return self.config_space
    
    def to_synetune(self):
        hyperparams = self.config_space.get_hyperparameters()
        synetune_config_dict = {}
        for hp in hyperparams:
            if isinstance(hp, UniformIntegerHyperparameter):
                synetune_config_dict[hp.name] = Integer(hp.lower, hp.upper)
            elif isinstance(hp, UniformFloatHyperparameter):
                # sample num_neighbors in 20% of unit ball around curr_value
                synetune_config_dict[hp.name] = Float(hp.lower, hp.upper)
            elif isinstance(hp, OrdinalHyperparameter):
                synetune_config_dict[hp.name] = Ordinal(hp.sequence)
        return synetune_config_dict
    
    def is_valid(self, config):
        return True