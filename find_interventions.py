"""
     This script is a helper to find good interventions for our experimental evaluation.
     To simulate user knowledge (good and bad) we randomly sample hyperparameter
     configurations from all search spaces and obtain the corresponding evaluation metrics.
     We then sort the evaluated configurations w.r.t. test-accuracy/regret and 
     consider the best/worst 50 samples. Among these 50 samples we identify we identify the
     K "purest" hyperparameters, i.e. the hyperparameters with least variation which
     led to the 50 best/worst evaluations.
     This should give us the most important hyperparameters and their corresponding value.
"""

from ihpo.benchmarks import HPOTabularBenchmark, NAS101Benchmark, NAS201Benchmark, JAHSBenchmark, TransNASBench, BenchQueryResult
import argparse
import pandas as pd 
import numpy as np
from typing import List
from sklearn.preprocessing import OrdinalEncoder

def build_dataframe(configs: List[dict], evaluations: List[BenchQueryResult]):
    df_dict = {'metric': []}
    for cfg, eval in zip(configs, evaluations):
        for k, v in cfg.items():
            if k not in df_dict:
                df_dict[k] = [v]
            else:
                df_dict[k].append(v)

        df_dict['metric'].append(eval.test_performance)
    return pd.DataFrame.from_dict(df_dict)

def get_intervention_suggestions(df: pd.DataFrame, N, t):
    df = df.sort_values(by='metric', ascending=False)
    numerical_cols = df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns.tolist()
    non_numerical_cols = [c for c in df.columns if c not in numerical_cols]

    # prepare all non-numerical columns for computation of entropy
    encoder = OrdinalEncoder()
    df[non_numerical_cols] = encoder.fit_transform(df[non_numerical_cols], df['metric'])
    bad_evals, good_evals = df[df['metric'] < t], df[df['metric'] >= t]
    if bad_evals.shape[0] > N:
        bad_evals = bad_evals[-N:]
    if good_evals.shape[0] > N:
        good_evals = good_evals[:N]

    most_freq_good = compute_hyperparameter_entropy(good_evals, non_numerical_cols)
    most_freq_bad = compute_hyperparameter_entropy(bad_evals, non_numerical_cols)

    print(f"GOOD; MAX: {good_evals['metric'].max()}, MIN: {good_evals['metric'].min()}")
    print(f"BAD; MAX: {bad_evals['metric'].max()}, MIN: {bad_evals['metric'].min()}")

    # reverse transform. Requires a little hack...
    good_config = most_freq_good['most_freq'].to_numpy().reshape(1, -1)
    bad_config = most_freq_bad['most_freq'].to_numpy().reshape(1, -1)

    non_numerical_cols_idx = [i for i, c in enumerate(df.columns) if c in non_numerical_cols]
    rev_good_non_numerical = encoder.inverse_transform(good_config[:, non_numerical_cols_idx]).flatten()
    rev_bad_non_numerical = encoder.inverse_transform(bad_config[:, non_numerical_cols_idx]).flatten()

    # now we can replace the encoded value of categorical hyperparameters with the
    # reverse transformed one
    for i, c in enumerate(non_numerical_cols):
        most_freq_good[most_freq_good['c'] == c]['most_freq'] = rev_good_non_numerical[i]
        most_freq_bad[most_freq_bad['c'] == c]['most_freq'] = rev_bad_non_numerical[i]

    most_freq_good = most_freq_good.sort_values(by='ent', ascending=True)
    most_freq_bad = most_freq_bad.sort_values(by='ent', ascending=True)

    return most_freq_good, most_freq_bad

def compute_hyperparameter_entropy(df, non_numerical_cols):
    # compute entropy of each hyperparameter
    # rationale: the higher the entropy of a hyperparameter given e.g. good results, the less important it is
    # as it can vary quite freely.
    entropy_column_df = {'ent': [], 'c': [], 'most_freq': []}
    for c in df.columns:
        if c in non_numerical_cols:
            bins = len(np.unique(df[c]))
        else:
            bins = max(2, int(len(np.unique(df[c])) / 10))
        hist, _ = np.histogram(df[c].to_numpy(), bins=bins, density=False)
        p = hist / len(df)
        ent = -np.sum(p*np.log2(p))
        entropy_column_df['ent'].append(np.round(ent, 3))
        entropy_column_df['c'].append(c)
        # get most frequent value of hyperparameter c (used for intervention then)
        unique_vals, counts = np.unique(df[c], return_counts=True)
        most_freq = unique_vals[np.argmax(counts)]
        entropy_column_df['most_freq'].append(most_freq)
    
    return pd.DataFrame.from_dict(entropy_column_df)

def get_benchmark(args):
    task = args.task
    benchmark = args.benchmark
    if benchmark == 'hpo':
        return HPOTabularBenchmark('xgb', 167120)
    elif benchmark == 'nas101':
        return NAS101Benchmark(task)
    elif benchmark == 'nas201':
        return NAS201Benchmark(task)
    elif benchmark == 'jahs':
        return JAHSBenchmark(task)
    elif benchmark == 'transnas':
        return TransNASBench(task)
    else:
        raise ValueError('No such benchmark')


parser = argparse.ArgumentParser()

parser.add_argument('--samples', default=500, type=int)
parser.add_argument('--benchmark')
parser.add_argument('--task')
parser.add_argument('--mode', default='random', choices=['random', 'entropy_based'])
parser.add_argument('--t', type=float, default=10)
parser.add_argument('--N', type=int, default=10)

args = parser.parse_args()

benchmark = get_benchmark(args)
configs = benchmark.search_space.sample(args.samples)
results = []
for cfg in configs:
    cfg['Optimizer'] = 'SGD'
    res = benchmark.query(cfg)
    results.append(res)

if args.mode == 'entropy_based':
    df = build_dataframe(configs, results)
    good_interventions, bad_interventions = get_intervention_suggestions(df, args.N, args.t)
    good_filename = f'./interventions/{args.benchmark}_{args.task}_good.csv'
    bad_filename = f'./interventions/{args.benchmark}_{args.task}_bad.csv'
    good_interventions.to_csv(good_filename)
    bad_interventions.to_csv(bad_filename)
else:
    performances = np.array([res.test_performance for res in results])
    best_idx, worst_idx = np.argwhere(performances == performances.max()).flatten()[0], np.argwhere(performances == performances.min()).flatten()[0]
    print("====================== RESULTS ===================")
    print(f"Best Performance: {performances[best_idx]}")
    print(f"Config: {configs[best_idx]}")
    print(f"Worst Performance: {performances[worst_idx]}")
    print(f"Config: {configs[worst_idx]}")