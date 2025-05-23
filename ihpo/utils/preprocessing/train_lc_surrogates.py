"""
    This script trains a surrogte model for each of the 35 datasets in the LC Benchmark.
    All models are stored in a path passed by the user, so that we can use the surrogates instead of
    loading the entire LC Bench data each time we evaluate an algorithm.
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import pickle
import argparse
import os

LC_HYPERPARAMETERS = ['num_layers', 'max_units', 'batch_size', 'learning_rate', 'momentum', 'max_dropout', 'weight_decay']

def load_json_data(path):
    with open(path, 'r') as fin:
        df = pd.read_json(fin, orient='records', lines=True)
    return df

def get_dataset_data(df: pd.DataFrame):
      """
        Build a dataframe with log-data from each task
      """
      train_metric_data_dict = {}
      val_metric_data_dict = {}
      test_metric_data_dict = {}

      for task in df.columns:
         print(f"\t =========== {task} =========")
         train_metric_data_per_budget = {}
         val_metric_data_per_budget = {}
         test_metric_data_per_budget = {}
         dictionary = df[task][0]
         for cfg_id in dictionary.keys():
              budgets = list(dictionary[cfg_id].keys())
              for b in budgets:
                config_and_result_dict = dictionary[cfg_id][b]
                config_dict = config_and_result_dict['config']
                results = config_and_result_dict['results']
                hp_vec = [config_dict[h] for h in LC_HYPERPARAMETERS]
                for seed in results.keys():
                    train_row = hp_vec + [results[seed]['final_train_balanced_accuracy']]
                    val_row = hp_vec + [results[seed]['final_val_balanced_accuracy']]
                    test_row = hp_vec + [results[seed]['final_test_balanced_accuracy']]
                    if b in train_metric_data_per_budget:
                        train_metric_data_per_budget[b].append(train_row)
                        val_metric_data_per_budget[b].append(val_row)
                        test_metric_data_per_budget[b].append(test_row)
                    else:
                        train_metric_data_per_budget[b] = [train_row]
                        val_metric_data_per_budget[b] = [val_row]
                        test_metric_data_per_budget[b] = [test_row]

         train_metric_data_dict[task] = train_metric_data_per_budget
         val_metric_data_dict[task] = val_metric_data_per_budget
         test_metric_data_dict[task] = test_metric_data_per_budget

      return train_metric_data_dict, val_metric_data_dict, test_metric_data_dict


def train_surrogates(metric_data_dict, metric_split, surr_path):
    # Split the dataset into training and testing sets
    for task, data_per_budget in metric_data_dict.items():
        print(f"\t =========== TASK: {task} ===========")
        for budget, data in data_per_budget.items():
            data = np.array(data)
            X, y = data[:, :-1], data[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize the Gradient Boosting Classifier
            gb_model = RandomForestRegressor(n_estimators=100, random_state=42)

            # Train the model
            gb_model.fit(X_train, y_train)

            # Make predictions
            y_pred = gb_model.predict(X_test)

            # Evaluate the model
            r2 = r2_score(y_test, y_pred)
            print(f"R2-score on {task}: {r2:.2f}")

            # save the model
            if not os.path.exists(os.path.join(surr_path, task)):
                os.mkdir(os.path.join(surr_path, task))
            with open(os.path.join(surr_path, task) + f'/model_{metric_split}_at_{budget}', 'wb') as f:
                pickle.dump(gb_model, f)


parser = argparse.ArgumentParser()

parser.add_argument('--data-path')
parser.add_argument('--surrogate-path')

args = parser.parse_args()

print("======== LOAD DATA =========")
df = load_json_data(args.data_path)
print("======== EXTRACT DATA ========")
train_metric_data, val_metric_data, test_metric_data = get_dataset_data(df)
print("======== FIT TRAIN ========")
train_surrogates(train_metric_data, 'train', args.surrogate_path)
print("======== FIT VAL ========")
train_surrogates(val_metric_data, 'val', args.surrogate_path)
print("======== FIT TEST ========")
train_surrogates(test_metric_data, 'test', args.surrogate_path)

