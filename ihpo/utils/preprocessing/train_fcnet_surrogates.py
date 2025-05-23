"""
    NOTE: The surrogates do not perform well on all datasets. Thus, we use FCNet as a regular tabular benchmark.
"""

import h5py
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import pickle
import argparse
import os
from pathlib import Path


FCNET_HYPERPARAMETERS = ['activation_fn_1','activation_fn_2', 'batch_size', 'dropout_1', 'dropout_2', 'init_lr', 'lr_schedule', 'n_units_1', 'n_units_2']
FCNET_METRICS = ['final_test_error', 'valid_mse']
FCNET_TASKS = ['naval', 'parkinsons', 'protein', 'slice_localization']
FCNET_DATA_FILES = ['fcnet_naval_propulsion_data', 'fcnet_parkinsons_telemonitoring_data', 'fcnet_protein_structure_data', 'fcnet_slice_localization_data']

def load_data(path):
    """
        Load data from hdf5 file and convert it into pandas dataframe.
        Apply label encoding on all categorical non-numerical features.
    """
    data = h5py.File(path)
    configs, te, ve = [], [], []
    for k in data.keys():
        configs.append(json.loads(k))
        te.append(np.mean(data[k]["final_test_error"]))
        ve.append(np.mean(data[k]["valid_mse"][:, -1]))
    
    dict_data = {key: [d[key] for d in configs] for key in configs[0]}
    config_df = pd.DataFrame.from_dict(dict_data)

    labels_to_encode = ['lr_schedule', 'activation_fn_1', 'activation_fn_2']

    for l in labels_to_encode:
        # Initialize LabelEncoder
        encoder = LabelEncoder()

        # Fit and transform
        config_df[l] = encoder.fit_transform(config_df[l].to_numpy())

    config_df['val_error'] = ve
    config_df['test_error'] = te
    return config_df

def train_surrogates(datasets, surr_path):
    """
        Train a set of surrogates given a list of tabular benchmarks.
    """
    for task, data in datasets:

        for tc in ['val_error', 'test_error']:
            X, y = data[FCNET_HYPERPARAMETERS].to_numpy(), data[tc].to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            gb_model.fit(X_train, y_train)
            # Make predictions
            y_pred = gb_model.predict(X_test)
            # Evaluate the model
            r2 = r2_score(y_test, y_pred)
            print(f"R2-score on {tc}: {r2:.2f}")

            model_path = os.path.join(surr_path, task)
        
            if not os.path.exists(model_path):
                path = Path(model_path)
                path.mkdir(exist_ok=True, parents=True)
            with open(model_path + f'/{tc}', 'wb') as f:
                pickle.dump(gb_model, f)



parser = argparse.ArgumentParser()

parser.add_argument('--data-path')
parser.add_argument('--surrogate-path')

args = parser.parse_args()

datasets = []
for task, file_name in zip(FCNET_TASKS, FCNET_DATA_FILES):
    data_file = os.path.join(args.data_path, file_name + '.hdf5')

    df = load_data(data_file)

    datasets.append((task, df))

train_surrogates(datasets, args.surrogate_path)