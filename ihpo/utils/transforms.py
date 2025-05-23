from ..search_spaces import SearchSpace
import torch
import numpy as np

DTYPES_TO_TRANSFORM = ['str', 'int', 'bool', 'cat']

class ConfigurationScaleOneHotTrnasform:

    def __init__(self, search_sapce: SearchSpace) -> None:
        self.search_space = search_sapce
        self._transform_meta_data = {}

    def transform_configuration(self, configuration: dict):
        """
            Transform a configuration into a GP-friendly representation.
            We do one-hot encoding here for discrete variables and min max scaling for continuous.
            This is the same we do in the preprocessing part for historical runs.
            However, note that for the target task, we initially don't have data to do normalization.
            Thus, we have to rely on the search space definition to do the transform.
        """

        space_def = self.search_space.get_search_space_definition()

        transformed_cfg = []

        transformed_idx_counter = 0

        for name, d in space_def.items():
            if d['dtype'] == 'str' or d['dtype'] == 'cat':
                # one hot encoding
                choices = d['allowed']
                cfg_val = configuration[name]
                idx = [i for i in range(0, len(choices)) if choices[i] == cfg_val][0] # if this fails, value was invalid!
                binary_vec = [0] * (len(choices) - 1)
                if idx < (len(choices) - 1):
                    binary_vec[idx] = 1
                transformed_cfg += binary_vec
                self._transform_meta_data[(transformed_idx_counter, transformed_idx_counter + len(binary_vec))] = {
                    'name': name,
                    'dtype': d['dtype'],
                    'choices': choices
                }
                transformed_idx_counter += len(binary_vec)

            elif d['dtype'] == 'bool':
                choices = d['allowed']
                cfg_val = configuration[name]
                transformed_cfg.append(int(cfg_val))
                self._transform_meta_data[(transformed_idx_counter,)] = {
                    'name': name,
                    'dtype': d['dtype'],
                    'choices': choices
                }
                transformed_idx_counter += 1

            elif d['dtype'] == 'float':
                # scale between 0 and 1
                mi, ma = d['min'], d['max']
                cfg_val = configuration[name]
                scaled_val = (cfg_val - mi) / (ma - mi)
                transformed_cfg.append(scaled_val)
                self._transform_meta_data[(transformed_idx_counter,)] = {
                    'name': name,
                    'dtype': d['dtype'],
                    'min': mi,
                    'max': ma
                }
                transformed_idx_counter += 1

            else:
                mi, ma = min(d['allowed']), max(d['allowed'])
                cfg_val = configuration[name]
                scaled_val = (cfg_val - mi) / (ma - mi)
                transformed_cfg.append(scaled_val)
                self._transform_meta_data[(transformed_idx_counter,)] = {
                    'name': name,
                    'dtype': d['dtype'],
                    'min': mi,
                    'max': ma
                }
                transformed_idx_counter += 1

        return transformed_cfg

    def inv_transform_configuration(self, configuration: torch.Tensor):
        """
            Invert transformation.

            :param configuration: torch tensor with one-hot encoded and scaled hyperparameters

            :return: Dictionary with configuration values
        """
        cfg = {}
        for idx, meta_info in self._transform_meta_data.items():
            name = meta_info['name']
            if meta_info['dtype'] == 'str' or meta_info['dtype'] == 'cat': 
                choices = meta_info['choices']
                one_hot_val = configuration[idx[0]:idx[1]].numpy().flatten()
                # NOTE: Acquisition function might give us real values although we have discrete data
                #   This can happen if e.g. a GP is used and the GP is defined over R^n
                #   In this case, we take the max as 1 and set rest 0
                if not np.all([x == one_hot_val[0] for x in one_hot_val]):
                    active_idx = np.argmax(one_hot_val).flatten()
                else:
                    active_idx = []
                if len(active_idx) == 0:
                    inv_transform_val = choices[-1]
                else:
                    inv_transform_val = choices[active_idx[0]]
                cfg[name] = inv_transform_val

            elif meta_info['dtype'] == 'bool':
                # NOTE: Acquisition function might give us real values although we have discrete data
                #   This can happen if e.g. a GP is used and the GP is defined over R^n
                #   In this case, we define 0.5 as the thershold to decide whether value is True or False
                transformed_val = configuration[idx].item()
                inv_transform_val = True if transformed_val >= 0.5 else 0
                cfg[name] = inv_transform_val

            else:
                scaled_idx = idx[0]
                val = configuration[scaled_idx].item()
                scale = meta_info['max'] - meta_info['min']
                inv_transform_val = val * scale + meta_info['min']
                if meta_info['dtype'] == 'int':
                    inv_transform_val = round(inv_transform_val, 0)
                cfg[name] = inv_transform_val
        return cfg
    

class ConfigurationNumericalTransform:

    def __init__(self, search_sapce: SearchSpace) -> None:
        self.search_space = search_sapce
        self._transform_meta_data = {}
        self._variables = list(self.search_space.get_search_space_definition())

        # initialize metadata and store which variables have to be transformed (with corresponding index)
        for key_idx, (key, val) in enumerate(self.search_space.get_search_space_definition().items()):
            if val['dtype'] in DTYPES_TO_TRANSFORM:
                self._transform_meta_data[key_idx] = (key, True)
            else:
                self._transform_meta_data[key_idx] = (key, False)
    
    def transform_configuration(self, configuration: dict, trgt_col_order=None):
        vector_repr = []

        for key_idx, (key, val) in enumerate(self.search_space.get_search_space_definition().items()):
            if val['dtype'] in DTYPES_TO_TRANSFORM:
                choices = val['allowed']
                idx = choices.index(configuration[key])
                vector_repr.append(idx)
            else:
                if val['is_log']:
                    vector_repr.append(np.log(configuration[key]))
                else:
                    vector_repr.append(configuration[key])

        if trgt_col_order is not None:
            vector_repr = reorder_vector_columns(vector_repr, self._variables, trgt_col_order)
    
        return vector_repr

    def inv_transform_configuration(self, config_vec: list, is_col_order=None):
        dict_repr = {}
        search_space_def = self.search_space.get_search_space_definition()

        if is_col_order is not None:
            config_vec = reorder_vector_columns(config_vec, is_col_order, self._variables)

        for idx, val in enumerate(config_vec):
            key, transformed = self._transform_meta_data[idx]
            if transformed:
                choices = search_space_def[key]['allowed']
                config_val = choices[int(val)]
                dict_repr[key] = config_val
            else:
                if search_space_def[key]['is_log']:
                    val = np.exp(val)
                dict_repr[key] = val
        return dict_repr
    
def reorder_vector_columns(vector: list, is_col_order: list, target_col_order: list):
    """Given a configuration vector over a set of variables ordered as in is_col_order, reorder its entries 
        according to target_col_order.

    Example:
        Given a vector [1, 0, 0] with column order [A, B, C],  the vector will be reordered into [0, 1, 0] 
        to match with the column order of target_col_order=[B, A, C].

    Args:
        vector (list): Vector representing a configuration
        is_col_order (list): Column order of vector
        target_col_order (list): Target order of columns of vector

    Returns:
        list: Reordered vector.
    """
    new_vector = []
    for name in target_col_order:
        vec_col_idx = is_col_order.index(name)
        new_vector.append(vector[vec_col_idx])
    return new_vector