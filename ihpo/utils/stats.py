import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import join
import argparse

def read_csvs(result_path):
    dfs = []
    files = listdir(result_path)
    print(result_path)
    if len(files) > 1000:
        # randomly sample 500 runs
        indices = np.random.choice(np.arange(len(files)), 100, replace=False)
        files = [files[i] for i in indices]
    for file in files:
        pth = join(result_path, file)
        try:
            df = pd.read_csv(pth, index_col=0)
            dfs.append(df)
        except pd.errors.EmptyDataError:
            continue

    return dfs

def compute_speedups(exp):
    prefix = f'../../data/logs/{exp}_'
    algos = ['pc_int', 'pc_int_early']
    ref_dfs = read_csvs(prefix + 'pc')

    performances = []
    costs = []
    for rdf in ref_dfs:
        mp = rdf['test_performance'].max()
        idx = rdf.index[rdf['test_performance'] == mp].tolist()
        cost = rdf.iloc[:idx[0]]['cost'].sum()
        costs.append(cost)
        performances.append(mp)
    avg_perf = np.mean(performances)
    avg_cost = np.mean(costs)

    speedup_dict = {}

    for algo in algos:
        path = prefix + algo
        dfs = read_csvs(path)

        speedups = []
        for df in dfs:

            best_result_idx = df.index[df['test_performance'] >= avg_perf].tolist()
            first_idx = best_result_idx[0]
            cost = df.iloc[:first_idx]['cost'].sum()

            if cost == 0:
                continue # skip 0 cost runs, something's wrong there

            speedup = avg_cost / cost
            speedups.append(speedup)
        
        speedup_dict[algo] = np.array(speedups)
    
    return speedup_dict

