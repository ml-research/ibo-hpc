"""
    This script trains a surrogte model for each of the 35 datasets in the LC Benchmark.
    All models are stored in a path passed by the user, so that we can use the surrogates instead of
    loading the entire LC Bench data each time we evaluate an algorithm.
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import pickle
import argparse
import os

PD1_HYPERPARAMETERS = ['hps.lr_hparams.decay_steps_factor', 'hps.lr_hparams.initial_value', 'hps.lr_hparams.power', 'hps.opt_hparams.momentum']
PD1_EVALUATION_METRICS = ['best_train/error_rate', 'best_valid/error_rate']
PD1_TASKS = [
    ('cifar10', 'wide_resnet', 256, None),
    ('cifar10', 'wide_resnet', 2048, None),
    ('cifar100', 'wide_resnet', 256, None),
    ('cifar100', 'wide_resnet', 2048, None),
    ('fashion_mnist', 'max_pooling_cnn', 256, 'relu'),
    ('fashion_mnist', 'max_pooling_cnn', 2048, 'relu'),
    ('fashion_mnist', 'max_pooling_cnn', 256, 'tanh'),
    ('fashion_mnist', 'max_pooling_cnn', 2048, 'tanh'),
    ('fashion_mnist', 'simple_cnn', 256, None),
    ('fashion_mnist', 'simple_cnn', 2048, None),
    ('imagenet', 'resnet', 256, None),
    ('imagenet', 'resnet', 512, None),
    ('lm1b', 'transformer', 2048, None),
    ('mnist', 'max_pooling_cnn', 256, 'relu'),
    ('mnist', 'max_pooling_cnn', 2048, 'relu'),
    ('mnist', 'max_pooling_cnn', 256, 'tanh'),
    ('mnist', 'max_pooling_cnn', 2048, 'tanh'),
    ('mnist', 'simple_cnn', 256, None),
    ('mnist', 'simple_cnn', 2048, None),
    ('svhn_no_extra', 'wide_resnet', 256, None),
    ('svhn_no_extra', 'wide_resnet', 1024, None),
    ('translate_wmt', 'xformer_translate', 64, None),
    ('uniref50', 'transformer', 128, None),
]

def load_json_data(path):
    matched_path = path + 'pd1_matched_phase1_results.jsonl'
    with open(matched_path, 'r') as fin:
        df1 = pd.read_json(fin, orient='records', lines=True)

    unmatched_path = './benchmark_data/pd1/pd1_unmatched_phase1_results.jsonl'
    with open(unmatched_path, 'r') as fin:
        df2 = pd.read_json(fin, orient='records', lines=True)

    df = pd.concat((df1, df2))
    return df


def train_surrogates(df, surr_path):
    """
        Function to fit surrogate models on the PD1 benchmark.
        As a surrogate, we use GradientBoosting models from sklearn with 100 estimators.
        Further, surrogates are cached if no path [file_path_to_PD1]/surrogates/ exists.
        If the path exists, the cached models are loaded instead of trained.
    """
    if not os.path.exists(surr_path):
        os.mkdir(surr_path)
    for ds, model, bs, act_fn in PD1_TASKS:
        if act_fn is not None:
            data = df[(df['dataset'] == ds) & (df['model'] == model) & (df['hps.batch_size'] == bs) & (df['hps.activation_fn'] == act_fn)]
        else:
            data = df[(df['dataset'] == ds) & (df['model'] == model) & (df['hps.batch_size'] == bs)]

        task = f"{ds}_{model}_{bs}_{act_fn}"
        metrics = PD1_EVALUATION_METRICS if ds != 'lm1b' else ['best_valid/error_rate', 'best_valid/error_rate'] # for LM1B no training data is available, use valid data.
        for i, tc in enumerate(metrics):
            data.loc[:, tc] = data[tc].fillna(data[tc].max()) # fill diverged runs with worst result seen
            X, y = data[PD1_HYPERPARAMETERS].to_numpy(), data[tc].to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            gb_model = RandomForestRegressor(n_estimators=100, random_state=42)
            gb_model.fit(X_train, y_train)
            # Make predictions
            y_pred = gb_model.predict(X_test)

            # Evaluate the model
            r2 = r2_score(y_test, y_pred)
            print(f"R2-score on {task}[{tc}]: {r2:.2f}")

            model_name = tc.split('/')[0]
            if ds == 'lm1b' and i == 1:
                model_name = 'best_train' # rename the model here because no train data for LM1B
            model_path = os.path.join(surr_path, task)
        
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            with open(model_path + f'/{model_name}', 'wb') as f:
                pickle.dump(gb_model, f)


parser = argparse.ArgumentParser()

parser.add_argument('--data-path')
parser.add_argument('--surrogate-path')

args = parser.parse_args()

print("======== LOAD DATA =========")
df = load_json_data(args.data_path)
print("======== FIT SURROGATES ========")
train_surrogates(df, args.surrogate_path)

