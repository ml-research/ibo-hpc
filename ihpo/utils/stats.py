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

def compute_slowdown_v2(exp):
    prefix = f'../../data/{exp}_'
    algos = ['pc']
    ref_dfs = read_csvs(prefix + 'pc_int_recover')

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

        slowdowns = []
        for df in dfs:

            best_result_idx = df.index[df['test_performance'] >= avg_perf].tolist()
            if len(best_result_idx) > 0:
                first_idx = best_result_idx[0]
                cost = df.iloc[:first_idx]['cost'].sum()
            else:
                cost = df['cost'].sum()

            if cost == 0:
                continue # skip 0 cost runs, something's wrong there

            slowdown = avg_cost / cost
            slowdowns.append(-slowdown)
        
        speedup_dict[algo] = np.array(slowdowns)
    
    return speedup_dict

def compute_slowdown(exp, postfix='pc_int_recover'):
    prefix = f'../../data/logs_v2/{exp}_'
    ref_dfs = read_csvs(prefix + postfix)

    pc_dfs = read_csvs(prefix + 'pc')
    perfs = []
    for df in pc_dfs:
        perfs.append(df['test_performance'].max())

    mean_perf = np.mean(perfs)
    costs = []
    for df in pc_dfs:
        achieved_perfs = df.index[df['test_performance'] >= mean_perf].tolist()
        if len(achieved_perfs) > 0:
            idx = min(achieved_perfs)
            costs.append(df.iloc[:idx]['cost'].sum())
    costs = np.array(costs)

    recovered_costs = []
    for rdf in ref_dfs:
        achieved_perfs = rdf.index[rdf['test_performance'] >= mean_perf].tolist()
        if len(achieved_perfs) > 0:
            idx = min(achieved_perfs)
            recovered_costs.append(rdf.iloc[:idx]['cost'].sum())
    recovered_costs = np.array(recovered_costs)
    mn, mx = min(min(costs), min(recovered_costs)), max(max(costs), max(recovered_costs))
    costs = (costs - mn) / (mx - mn)
    recovered_costs = (recovered_costs - mn) / (mx - mn)

    catch_up_frac = len(recovered_costs) / len(ref_dfs)
    print(f"Catch-up: {catch_up_frac}; No catch-up: {1 - catch_up_frac}")
    
    mean_cost = np.mean(costs)
    slowdown = recovered_costs / mean_cost
    
    return np.median(slowdown), np.std(slowdown)

def compute_no_catch_up_diff(exp):
    prefix = f'../../data/{exp}_'
    ref_dfs = read_csvs(prefix + 'bopro_recover')

    pc_dfs = read_csvs(prefix + 'bopro')
    perfs = []
    for df in pc_dfs:
        perfs.append(df['test_performance'].max())

    mean_perf = np.mean(perfs)

    non_recovered_perfs = []
    for rdf in ref_dfs:
        achieved_perfs = rdf.index[rdf['test_performance'] >= mean_perf].tolist()
        if len(achieved_perfs) > 0:
            performance = rdf['test_performance'].max()
            non_recovered_perfs.append(performance)
    
    print(f"FRAC REACHED: {len(non_recovered_perfs) / len(ref_dfs)}")
            
    non_recovered_perfs = np.array(non_recovered_perfs)
    print(f"AVG NO INTERACTION: {mean_perf}")
    print(f"STD NO INTERACTION: {np.std(perfs)}")
    print(f"AVG INTERACTION: {np.mean(non_recovered_perfs)}")
    print(f"STD INTERACTION: {np.std(non_recovered_perfs)}")
    return non_recovered_perfs, np.array(perfs)