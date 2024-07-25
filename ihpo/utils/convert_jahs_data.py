"""
    This script converts the hyperparameter definitions stored in pickle files
    provided by JAHS to JSON format. This has to be done because the jahs_bench
    package relies on ConfigSpace==0.4.21 while ihpo relies on ConfigSpace>=0.7.1
    which are not compatible. 

    NOTE: To run this script please create a new environment with the original jahs_bench
    installed (not the one we provide due to compatibility issues).

    NOTE: When executing this script on freshly downloaded benchmark files, writing the
    JSON file will fail due to permission issues. To cirumvent this, you can set the
    permissions appropriately using `sudo chmod -R 777 ./data/assembled_surrogates/`.
    It's recommended to reset the permissions afterwards.
"""
import json
import joblib
import os
try:
    from ConfigSpace.hyperparameters import UniformFloatHyperparameter, OrdinalHyperparameter, CategoricalHyperparameter
except ImportError:
    print("Failed to import hyperparameters from ConfigSpace. Make sure to use ConfigSpace==0.4.21")
    exit(-1)

def convert_surrogate_data(path):

    for dataset_storage in os.listdir(path):
        # iterate over tasks
        for metric_storage in os.listdir(os.path.join(path, dataset_storage)):
            pkl_file = os.path.join(path, dataset_storage, metric_storage, 'params.pkl.gz')
            params = joblib.load(pkl_file)
            json_dict = {}
            for k, v in params.items():
                if k == 'config_space':
                    configs = []
                    for hp in v.get_hyperparameters():
                        hp_dict = _convert_config_space(hp)
                        configs.append(hp_dict)
                    json_dict[k] = configs
                elif k == 'label_headers':
                    json_dict[k] = list(v.values)
                elif k == 'feature_headers':
                    json_dict[k] = list(v.values)
                else:
                    json_dict[k] = v
            json_file = os.path.join(path, dataset_storage, metric_storage, 'params.json')
            with open(json_file, 'w+', encoding='utf-8') as f:
                json.dump(json_dict, f)

def _convert_config_space(hp):
    hp_dict = {}
    if isinstance(hp, UniformFloatHyperparameter):
        hp_dict['type'] = 'UniformFloatHyperparameter'
        hp_dict['name'] = hp.name
        hp_dict['lower'] = hp.lower
        hp_dict['upper'] = hp.upper
        hp_dict['log'] = hp.log
    elif isinstance(hp, OrdinalHyperparameter):
        hp_dict['type'] = 'OrdinalHyperparameter'
        hp_dict['name'] = hp.name
        hp_dict['sequence'] = hp.sequence
        hp_dict['default_value'] = hp.default_value
    elif isinstance(hp, CategoricalHyperparameter):
        hp_dict['type'] = 'CategoricalHyperparameter'
        hp_dict['name'] = hp.name
        hp_dict['choices'] = hp.choices
        hp_dict['default_value'] = hp.default_value
    return hp_dict


convert_surrogate_data('../data/assembled_surrogates/')
