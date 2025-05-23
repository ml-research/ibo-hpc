import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os import listdir
from os.path import join
import argparse
from stats import compute_speedups, compute_slowdown, compute_no_catch_up_diff

plt.rcParams.update({'font.size': 20, 'font.family':'serif'})
#plt.rcParams.update(bundles.icml2024(family="serif", usetex=False, column="full", nrows=1))

BEST_PERFS = {
        'nas101': 1, # 0.9452,
        'nas201': 1, #0.945,
        'jahs_cifar10': 1.0,
        'jahs_co': 1.0,
        'jahs_fm': 1.0,
        'nas201_trans_cifar10': 0.93,
        'nas201_trans_cifar100': 0.74,
        'hpob_6767_31': 0.77,
        'hpob_6794_31': 0.78,
        'hpob_4796_homo': 0.6,
        'lc_adult': 1,
        'pd1_cifar10_wide_resnet_256_None': 1,
        'pd1_imagenet_resnet_256_None': 1,
        'toy_branin': 0,
        'rosenbrock_v3': 0,
        'quadratic': 0
    }

BENCHMARK_LABELS = {
        'nas101': 'NAS-Bench-101',
        'nas201': 'NAS-Bench-201',
        'jahs_cifar10': 'JAHS (CIFAR-10)',
        'jahs_co': 'JAHS (CO)',
        'jahs_fm': 'JAHS (Fashion-MNIST)',
        'branin': 'Branin',
        'rosenbrock_v3': 'Rosenbrock',
        'quadratic': 'Quadratic',
        'hpob_6767_31': 'HPOB 1',
        'hpob_6794_3': 'HPOB 2',
        'lc_adult': 'LCBench',
        'pd1_cifar10_wide_resnet_256_None': 'PD1',
        'pd1_imagenet_resnet_256_None': 'PD1'
    }

ALGO_COLORS = {
        'pc': ('#3492eb', 'solid'),
        'pc_trans': ('#34ebe5', 'solid'),
        'pc_trans_rf': ('#ea5cfa', 'solid'),
        'pc_int': ('#34ebe5', 'solid'),
        'pc_int_autorecover': ('#ea5cfa', 'solid'),
        'pc_int_recover': ('#ea5cfa', 'solid'),
        'pc_int_multi': ('#8a4000', 'solid'),
        'pc_int_many': ('#8a4000', 'solid'),
        'pc_int_early': ('#fa675c', 'solid'),
        'pc_int_dist': ('#15a123', 'solid'),
        'pc_new': ('#3492eb', 'solid'),
        'pc_int_new': ('#34ebe5', 'solid'),
        'pc_int_late': ('#34ebe5', 'solid'),
        'pc_int_multi_new': ('#8a4000', 'solid'),
        'pc_int_early_new': ('#fa675c', 'solid'),
        'pc_ablation_cond_05': ('#8a4000', 'solid'),
        'pc_ablation_cond_025': ('#fa675c', 'solid'),
        'pc_ablation_cond_075': ('#15a123', 'solid'),
        'pc_ablation_sample_size_5': ('#8a4000', 'solid'),
        'pc_ablation_sample_size_10': ('#fa675c', 'solid'),
        'pc_ablation_sample_size_30': ('#15a123', 'solid'),
        'pc_ablation_decay_03': ('#8a4000', 'solid'),
        'pc_ablation_decay_07': ('#fa675c', 'solid'),
        'pc_ablation_decay_099': ('#15a123', 'solid'),
        'gp': ('#8a4000', 'dashed'),
        'optunabo': ('#56048c', 'dashed'),
        'skoptbo': ('#B82132', 'dashed'),
        'pibo': ('#014d08', 'dotted'),
        'smac': ('#ed6618', 'dashed'),
        'rs': ('#964aed', 'dashed'),
        'rs_int_early': ('#344CB7', 'dotted'),
        'rs_int_late': ('#FF0080', 'dotted'),
        'ls': ('#cf06aa', 'dashed'),
        'priorband': ('#3B0000', 'dotted'),
        'bopro': ('#d10808', 'dotted'),
        'bopro_plain': ('#d10808', 'dotted'),
        'pibo_plain': ('#014d08', 'dotted'),
        'bopro_recover': ('#d10808', 'dotted'),
        'pibo_recover': ('#014d08', 'dotted'),
        'mphd_trans': ('#d10808', 'dashed'),
        'bbox_trans': ('#964aed', 'dashed'),
        'quant_trans': ('#ed6618', 'dashed'),
        'fsbo_trans': ('#ea5cfa', 'dashed'),
        'transbo_trans': ('#8a4000', 'dashed'),
        'rgpe_trans':  ('#15a123', 'dashed'),
        '0shot_trans': ('#cf06aa', 'dashed'),
        'ablr': ('#fa675c', 'dashed'),
    }


ALGO_NAME_MAPPING = {
        'pc': 'BO w/ PCs',
        'pc_trans': 'HyTraLVIP',
        'pc_trans_rf': 'HyTraLVIP w/ EI',
        'pc_int': 'IBO (w/ interaction@10)',
        'pc_int_early': 'IBO (w/ interaction@5)',
        'pc_int_multi': 'IBO (w/ interaction@5,20)',
        'pc_int_many': 'IBO (w/ 4 interactions)',
        'pc_int_autorecover': 'IBO (w/ interaction@5)',
        'pc_int_recover': 'IBO (w/ interaction@5)',
        'pc_int_dist': 'IBO (w/ dist. intervention@5)',
        'pc_new': 'IBO',
        'pc_int_new': 'IBO (w/ interaction@10)',
        'pc_int_late': 'IBO (w/ interaction@10)',
        'pc_int_early_new': 'IBO (w/ interaction@5)',
        'pc_int_multi_new': 'IBO (w/ interaction@5,20)',
        'pc_ablation_cond_05': 'IBO q=0.5',
        'pc_ablation_cond_025': 'IBO q=0.25',
        'pc_ablation_cond_075': 'IBO q=0.75',
        'pc_ablation_sample_size_5': 'IBO @ 5 samples',
        'pc_ablation_sample_size_10': 'IBO @ 10 samples',
        'pc_ablation_sample_size_30': 'IBO @ 30 samples',
        'pc_ablation_decay_03': 'IBO $\gamma$=0.3',
        'pc_ablation_decay_07': 'IBO $\gamma$=0.7',
        'pc_ablation_decay_099': 'IBO $\gamma$=0.99',
        'gp': 'BO w/ GP',
        'priorband': 'Priorband',
        'optunabo': 'BO w/ TPE',
        'skoptbo': 'BO w/ RF',
        'pibo': 'πBO',
        'smac': 'SMAC',
        'rs': 'RS',
        'rs_int_early': 'RS (w/ interaction@5)',
        'rs_int_late': 'RS (w/ interaction@10)',
        'rs_test': 'RSL',
        'ls': 'LS',
        'hyperband': 'Hyperband',
        'smac_test': 'SMAC',
        'hyperband_test': 'Hyperband',
        'bopro': 'BOPrO',
        'bopro_plain': 'BOPrO',
        'pibo_plain': 'PiBO',
        'bopro_recover': 'BOPrO Recover',
        'pibo_recover': 'PiBO Recover',
        'mphd_trans': 'MPHD',
        'bbox_trans': 'BBOX',
        'quant_trans': 'Quantile',
        'fsbo_trans': 'FSBO',
        'transbo_trans': 'TransBO',
        'rgpe_trans':  'RGPE',
        '0shot_trans': 'ZeroShot',
        'ablr': 'ABLR'
    }

TRANSFER_TASK_COLOR = {
    'nas201_trans' : {
        'cifar10': 'red',
        'cifar100': 'blue'
    }
}

TRANSFER_ALGO_LINESTYLE = {
    'rs': 'dotted',
    'pc': 'dashed',
    'pc_deactivated': 'dotted'
}

def read_csvs(result_path, max_reads=500):
    dfs = []
    files = listdir(result_path)
    print(result_path)
    if len(files) > max_reads:
        # randomly sample 500 runs
        indices = np.random.choice(np.arange(len(files)), max_reads, replace=False)
        files = [files[i] for i in indices]
    for file in files:
        pth = join(result_path, file)
        try:
            df = pd.read_csv(pth, index_col=0)
            dfs.append(df)
        except pd.errors.EmptyDataError:
            continue

    return dfs

def compute_regrets(result_path, best_perf, align_cost, num_exps=500, uncertainty='SE', is_transfer_exp=False):
    dfs = read_csvs(result_path, num_exps)
    test_perfs = []
    costs = []
    max_iters = int(1e9)

    for df in dfs:
        # drop invalid columns
        df = df[df['cost'] != float('inf')]
        if is_transfer_exp:
            if 'task' in df.columns:
                last_task = df.iloc[-1]['task']
                df = df[df['task'] == last_task]
        if df['cost'].isna().all():
            df.loc[:, 'cost'] = 1
        max_iters = min(max_iters, df.index.max())
        test_perf_per_iter = []
        cum_cost = []
        for i in range(0, max_iters):
            iterdf = df.iloc[:i]
            #iterdf = df.head(i)
            if i == 0 and align_cost:
                for j in range(iterdf.shape[0]):
                    tmp_df = iterdf[:j]
                    best = float(tmp_df['test_performance'].max())
                    regret = best_perf - (best / 100.)
                    test_perf_per_iter.append(regret)
                    np_cost = tmp_df['cost'].to_numpy()
                    cum_cost.append(np_cost[:j].sum())
            else:
                # get best result for that iteration
                best = float(iterdf['test_performance'].max())
                regret = best_perf - (best / 100.)
                cost = float(iterdf['cost'].sum())
                test_perf_per_iter.append(regret)
                cum_cost.append(cost)

        test_perfs.append(np.array(test_perf_per_iter))
        costs.append(np.array(cum_cost))
    
    # some algos don't run the full number of iterations in each execution (e.g. SMAC)
    # hence we cannot create an np.array -> go over each iteration and compute mean and std.
    results = []
    for i in range(max_iters - 2):
        perfs = [run_res[i] for run_res in test_perfs if (i - 1) < len(run_res)]
        rc = [run_cost[i] for run_cost in costs if (i-1) < len(run_cost)]
        perf_mean, perf_std = np.mean(perfs), np.std(perfs)
        if uncertainty == 'SE':
            perf_std = perf_std / np.sqrt(num_exps)
        mean_cost = np.mean(rc)
        results.append([perf_mean, perf_std, mean_cost])
    return np.array(results)

def collect_results(exp='nas101', num_exps=500, uncertainty='SE', is_transfer=False):
    prefix = f'../../data/new-exp/{exp}_'
    #algos = [ 'pc', 'pc_int_recover', 'pc_int_multi', 'rs', 'pibo', 'bopro', 'ls', 'smac']
    #algos = [ 'pc_ablation_cond_025', 'pc_ablation_cond_05', 'pc_ablation_cond_075']
    #algos = [ 'pc_ablation_sample_size_5', 'pc_ablation_sample_size_10', 'pc', 'pc_ablation_sample_size_30']
    #algos = [ 'pc_ablation_decay_03', 'pc_ablation_decay_07', 'pc_ablation_decay_099']
    #algos = ['fsbo_trans', '0shot_trans', 'quant_trans', 'pc_trans', 'rgpe_trans', 'mphd_trans', 'transbo_trans', 'bbox_trans', 'optunabo']
    #algos = ['mphd_trans', 'pc_trans', 'pc', 'optunabo', 'gp']
    #algos = ['pc', 'pc_int_recover', 'pc_int_many', 'pibo_recover', 'bopro_recover', 'smac', 'skoptbo', 'optunabo']
    #algos = ['pc', 'pc_int_late', 'pc_int_early', 'pc_int_dist', 'rs_int_late', 'rs_int_early', 'pibo', 'smac', 'optunabo', 'bopro', 'skoptbo', 'priorband']
    algos = ['pc', 'pc_int_late', 'pc_int_early', 'pc_int_dist', 'pibo', 'smac', 'optunabo', 'bopro', 'skoptbo', 'priorband', 'ls']
    result_paths = [prefix + algo for algo in algos]
    align_costs = lambda algo: algo.startswith('pc') or algo == 'ls'
    regrets = [(ALGO_NAME_MAPPING[algo], algo, compute_regrets(rp, BEST_PERFS[exp], align_costs(algo), num_exps, uncertainty, is_transfer)) for algo, rp in zip(algos, result_paths)]
    return regrets

def plot_regrets(regret_results, exp, file='./out.pdf', with_std=False):
    plt.rcParams.update({'font.size': 30, 'font.family':'serif'})
    fig = plt.figure(figsize=(8, 7))
    scales = {
        #'jahs_fm': ((0, 0.9e8), (0.048, 0.115)),
        'jahs_fm': ((0, 0.55e8), (0.048, 0.115)),
        #'nas101': ((0, 1.5e6), (0.055, 0.07)),
        'nas101': ((0, 1.25e6), (0.055, 0.064)),
        'nas201': ((0, 13000), (0.084, 0.1)),
        #'jahs_cifar10': ((0, 0.3e8), (0.07, 0.3)),
        'jahs_cifar10': ((0, 0.57e8), (0.07, 0.25)),
        #'jahs_co': ((0, 0.2e6), (0.05, 0.135)),
        'jahs_co': ((0, 0.35e7), (0.052, 0.125)),
        'hpob_6767_31': [(0, 90), (0.0, 0.1)],
        'hpob_6794_31': [(0, 90), (0.004, 0.011)],
        'hpob_4796_homo': [(0, 20), (0.59, 0.6)],
        'toy_branin': [(0, 2000), (-4, -1)],
        'rosenbrock_v3': [(0, 20), (0, 200)],
        'quadratic': [(0, 100), (0, 8.)],
        'lc_adult': [(0, 100), (0, 1)],
        'pd1_cifar10_wide_resnet_256_None': [(0, 100), (0, 1)],
        'pd1_imagenet_resnet_256_None': [(0, 100), (0, 1)]
    }
    for i, (algo, algo_id, res) in enumerate(regret_results):
        mean_perf, perf_std, mean_cost = res[:, 0], res[:, 1], res[:, 2]
        alpha = 1.0 if 'pc' in algo_id else 0.35
        c, line_type = ALGO_COLORS[algo_id]
        plt.plot(mean_cost, mean_perf, label=algo, alpha=alpha, linewidth=4., c=c, linestyle=line_type)
        if with_std:
            plt.fill_between(mean_cost, mean_perf-perf_std, mean_perf+perf_std, alpha=0.1, color=c)
    plt.xlabel('wall-clock time (sec.)')
    plt.ylabel('test error')
    #plt.yscale('log')
    #plt.xscale('log')
    if exp in scales.keys():
        xlim, ylim = scales[exp]
        plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend(fontsize=16)
    plt.grid(alpha=0.6)
    plt.savefig(file, bbox_inches='tight')

def plot_avg_regrets_per_iter(exp, file):
    """
        Plot the average regret in each iteration (not the best until here)
    """
    prefix = f'../../data/logs/{exp}_pc_int_'
    search_spaces = ['multi']
    search_space_regrets = {}
    for space in search_spaces:
        path = prefix + space
        dfs = read_csvs(path)

        # for each df, compute regrets at each sample
        regrets = []
        for df in dfs:
            rgr = []
            for i in np.unique(df['iter']):
                r = df[df['iter'] == i]['test_performance'].to_numpy().mean() / 100.
                rgr.append(BEST_PERFS[exp] - r)
            regrets.append(np.array(rgr))
        regrets = np.column_stack(regrets)
        regrets = np.mean(regrets, axis=1)
    
        search_space_regrets[space] = regrets
    
    colors = {
        's1': '#3492eb',
        's2': '#34ebe5',
        's3': '#8a4000',
        'multi': '#3492eb'
    }
    fig = plt.figure()
    for space, rgr in search_space_regrets.items():
        plt.plot(np.arange(len(rgr)), rgr, label=space, linewidth=2, c=colors[space])

    plt.xlabel('Iteration')
    plt.ylabel('Test error')

    plt.legend()
    plt.grid()
    plt.savefig(file)

def plot_cost_vs_optim_cost(file):
    base_path = '../../data/logs_v2/'
    exps = ['nas101', 'nas201', 'jahs_cifar10', 'jahs_co', 'jahs_fm']
    eval_perc_runtimes_mean, optim_perc_runtimes_mean = [], []
    eval_runtimes_stds, optim_runtimes_std = [], []
    eval_runtime_mean, optim_runtime_mean = [], []
    labels = []
    for e in exps:
        path = f'{base_path}{e}_pc_runtime/'
        dfs = read_csvs(path)
        eval_rts, optim_rts = [], []
        for df in dfs:
            eval_rts.append(df['cost'].sum())
            optim_rts.append(df['optim_cost'][0])
        
        mean_eval_rt = np.mean(eval_rts)
        mean_optim_rt = np.mean(optim_rts)
        eval_runtime_mean.append(mean_eval_rt)
        optim_runtime_mean.append(mean_optim_rt)
        eval_perc = mean_eval_rt / (mean_eval_rt + mean_optim_rt)
        optim_perc = mean_optim_rt / (mean_eval_rt + mean_optim_rt)
        eval_perc_runtimes_mean.append(eval_perc)
        optim_perc_runtimes_mean.append(optim_perc)
        eval_runtimes_stds.append(np.std(eval_rts))
        optim_runtimes_std.append(np.std(optim_rts))
        labels.append(BENCHMARK_LABELS[e])

    plt.rcParams.update({'font.size': 30, 'font.family':'serif'})
    fig = plt.figure(figsize=(13, 10))
    
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(labels))
    r2 = [x + bar_width for x in r1]

    # Create the bar plot
    plt.bar(r1, eval_perc_runtimes_mean, color='#51829B', width=bar_width, edgecolor='grey', label='Evaluation time (%)')
    plt.bar(r2, optim_perc_runtimes_mean, color='#C1E1C1', width=bar_width, edgecolor='grey', label='Optimization time (%)')
    print(f"Eval Runtime: {eval_runtime_mean} +/- {eval_runtimes_stds}")
    print(f"Optim Runtime: {optim_runtime_mean} +/- {optim_runtimes_std}")

    # Add labels and title
    plt.xlabel('Benchmark')
    plt.ylabel('Training vs. Search time (in %)')
    #plt.title('Bar Plot with Two Bars per Label')
    plt.xticks([r + bar_width / 2 for r in range(len(labels))], ['NAS-101 \n (C-10)', 'NAS-201 \n (C-10)', 'JAHS \n (C-10)', 'JAHS \n (FM)', 'JAHS \n(CH)'])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '20', '40', '60', '80', '100'])

    for i in range(len(eval_perc_runtimes_mean)):
        num1 = eval_perc_runtimes_mean[i] 
        num2 = optim_perc_runtimes_mean[i]
        plt.text(r1[i] - .15, num1 + .01, s=f'{num1 * 100:.2f}')
        plt.text(r2[i] - .15, num2 + .01, s=f'{num2 * 100:.2f}')

    # Add a legend
    plt.legend(loc='center right')

    plt.savefig(file, bbox_inches='tight')

def plot_runtime_comparison(file, algos):
    base_path = '../../data/logs_v2/'
    exps = ['nas201', 'nas101', 'jahs_cifar10', 'jahs_co', 'jahs_fm']
    pc_runtimes_mean, smac_runtimes_mean = [], []
    pc_rt_stds, smac_rt_stds = [], []
    labels = []
    for e in exps:
        pc_path = f'{base_path}{e}_pc_runtime/'
        smac_path = f'{base_path}{e}_smac_runtime/'
        pc_dfs = read_csvs(pc_path, 20)
        smac_dfs = read_csvs(smac_path, 20)
        pc_cost, smac_cost = [], []
        for pc_df, smac_df in zip(pc_dfs, smac_dfs):
            pc_cost.append(pc_df['optim_cost'][0])
            smac_cost.append(smac_df['optim_cost'][0])

        
        min_factor = min(min(pc_cost), min(smac_cost))
        max_factor = max(max(pc_cost), max(smac_cost))
        pc_cost, smac_cost = np.array(pc_cost), np.array(smac_cost)
        pc_cost_norm = pc_cost  (pc_cost - min_factor) / (max_factor - min_factor)
        smac_cost_norm = smac_cost (smac_cost - min_factor) / (max_factor - min_factor)
        
        cum_pc_cost = np.mean(pc_cost_norm)
        cum_smac_cost = np.mean(smac_cost_norm)
        pc_runtimes_mean.append(cum_pc_cost)
        smac_runtimes_mean.append(cum_smac_cost)
        pc_rt_stds.append(np.std(pc_cost_norm))
        smac_rt_stds.append(np.std(smac_cost_norm))
        labels.append(BENCHMARK_LABELS[e])

    plt.rcParams.update({'font.size': 30, 'font.family':'serif'})
    fig = plt.figure(figsize=(12, 10))
    
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(labels))
    r2 = [x + bar_width for x in r1]

    # Create the bar plot
    plt.bar(r1, pc_runtimes_mean, yerr=pc_rt_stds, color='#51829B', width=bar_width, edgecolor='grey', label='IBO-HPC runtime (%)')
    plt.bar(r2, smac_runtimes_mean, yerr=smac_rt_stds, color='#C1E1C1', width=bar_width, edgecolor='grey', label='SMAC runtime (%)')
    print(f"IBO Runtime: {pc_runtimes_mean} +/- {pc_rt_stds}")
    print(f"SMAC Runtime: {smac_runtimes_mean} +/- {smac_rt_stds}")

    # Add labels and title
    plt.xlabel('Benchmark (ordered by #hyperparameters)')
    plt.ylabel('Relative Runtime IBO-HPC vs. SMAC')
    #plt.title('Bar Plot with Two Bars per Label')
    plt.xticks([r + bar_width / 2 for r in range(len(labels))], ['NAS-201 \n (C-10)', 'NAS-101 \n (C-10)', 'JAHS \n (C-10)', 'JAHS \n (FM)', 'JAHS \n(CH)'])

    # Add a legend
    plt.legend()

    plt.savefig(file, bbox_inches='tight')

def plot_interaction_gain(file, version):
    plt.rcParams.update({'font.size': 30, 'font.family':'serif'})
    search_spaces = ['s1', 's2', 's3']

    benchmarks = ['jahs_cifar10', 'jahs_co', 'jahs_fm']

    search_space_sizes = {
        'v2': {
            's1': (1 - 1e-3) * (1e-2 - 1e-5) * 3 * 2 * 5**6 * 16 * 32 * 200 * 1,
            's2': (1 - 1e-3) * (1e-2 - 1e-5) * 3 * 2 * 5**6 * 8 * 14 * 200 * 1,
            's3': (1 - 1e-3) * (1e-2 - 1e-5) * 3 * 2 * 5**6 * 3 * 3 * 200 * 1,
        },
        'v3': {
            's1': (1 - 1e-3) * (1e-2 - 1e-5) * 3 * 2 * 5**6 * 16 * 32 * 200 * 1,
            's2': (1 - 1e-3) * (1e-2 - 1e-5) * 3 * 2 * 5**6 * 8 * 14 * 200 * 1,
            's3': (1 - 1e-3) * (1e-2 - 1e-5) * 3 * 2 * 5**6 * 3 * 3 * 200 * 1
        }
    }

    colors = {
        'nas101': '#32a852',
        'nas201': '#c98b2e',
        'jahs_cifar10': '#c9462e',
        'jahs_co': '#2e6fc9',
        'jahs_fm': '#a82ec9'
    }
    diffs = {}
    for b in benchmarks:
        prefix = f'../../data/search-space-size-{version}/{b}_pc_'
        space_diffs = []
        space_stds = []
        for space in search_spaces:
            vanilla_run_pth = prefix + space
            interaction_run_pth = prefix + 'int_' + space
            vanilla_dfs = read_csvs(vanilla_run_pth, 50)
            interaction_dfs = read_csvs(interaction_run_pth, 50)

            v_best = np.array([df['test_performance'].max() / 100. for df in vanilla_dfs])
            i_best = np.array([df['test_performance'].max() / 100. for df in interaction_dfs])
            diff = i_best[:46] - v_best[:46]
            avg_diff = np.average(diff)
            #avg_diff = - np.log(avg_diff / search_space_sizes[version][space])
            std_diff = np.std(diff)
            space_diffs.append(avg_diff)
            space_stds.append(std_diff)
        diffs[b] = (space_diffs, space_stds)

    bar_width = 0.2
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111)
    x = np.arange(3)
    y_max = 0
    for i, b in enumerate(benchmarks):
        space_diffs, space_std = diffs[b]
        y_max = max(y_max, max(space_diffs))
        c = colors[b]
        x_ = x + (i * bar_width)
        ax.bar(x_, space_diffs, color=c, label=BENCHMARK_LABELS[b], width=bar_width, yerr=space_std)

    ax.set_xticks([0.2, 1.2, 2.2], ['Large', 'Medium', 'Small'])

    plt.xlabel('Search Space Size')
    plt.ylabel('Accuracy Gain by Interaction')
    plt.ylim(-5e-3, y_max + 0.03)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(file, bbox_inches='tight')

def plot_speedups(file):
    plt.rcParams.update({'font.size': 30, 'font.family':'serif'})
    #plt.rc('axes', titlesize=20)
    exps = ['nas101', 'nas201', 'jahs_cifar10', 'jahs_fm', 'jahs_co']

    #pc_int_means, pc_int_early_means = [], []
    #pc_int_stds, pc_int_early_stds = [], []
    #for exp in exps:
    #    speedups = compute_speedups(exp)
    #    pc_int_means.append(np.mean(speedups['pc_int']))
    #    pc_int_early_means.append(np.mean(speedups['pc_int_early']))
    #    pc_int_early_stds.append(np.std(speedups['pc_int_early']))
    #    pc_int_stds.append(np.std(speedups['pc_int']))

    pc_int, pc_int_early = [], []
    for exp in exps:
        speedups = compute_speedups(exp)
        pc_int.append(speedups['pc_int'])
        pc_int_early.append(speedups['pc_int_early'])
    
    br1 = np.arange(len(pc_int)) 
    br2 = [x - (0.25+0.1) for x in br1]

    fig, ax = plt.subplots(figsize=(13, 10))

    #plt.bar(br1, pc_int_means, width=0.25, color='r', label='Interaction@5', yerr=pc_int_stds)
    #plt.bar(br2, pc_int_early_means, width=0.25, color='b', label='Interaction@10', yerr=pc_int_early_stds)

    bp1 = ax.boxplot(pc_int, positions=br1, widths=0.25, showfliers=False, boxprops={'linewidth': 4},)
    bp2 = ax.boxplot(pc_int_early, positions=br2, widths=0.25, showfliers=False, boxprops={'linewidth': 4})

    for k, v in bp1.items():
        if k not in ['whiskers', 'fliers', 'caps']:
            plt.setp(bp1.get(k), color='#FA8072')

    for k, v in bp2.items():
        if k not in ['whiskers', 'fliers', 'caps']:
            plt.setp(bp2.get(k), color='#7B3F00')
    #for p in bp1['boxes']:
    #    p.set_markerfacecolor('cyan')
#
    #for p in bp2['boxes']:
    #    p.set_markerfacecolor('tan')
    tick_pos = [(br1[i] + br2[i]) / 2 for i in range(len(br1))]
    plt.xticks(tick_pos, ['NAS-101 \n (C-10)', 'NAS-201 \n (C-10)', 'JAHS \n (C-10)', 'JAHS \n (FM)', 'JAHS \n(CH)'], size=25)

    plt.ylabel('Speedup')
    plt.xlabel('Benchmark')
    plt.ylim(-2, 20)
    plt.grid(axis='y', alpha=0.5)
    plt.ylim(0.0, 18.)

    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Interaction@10', 'Interaction@5'], loc='upper left')

    plt.savefig(file, bbox_inches='tight')

def plot_slowdown(file):
    plt.rcParams.update({'font.size': 30, 'font.family':'serif'})
    #plt.rc('axes', titlesize=20)
    exps = ['nas101', 'nas201', 'jahs_cifar10', 'jahs_fm', 'jahs_co']

    results_slowdown = []
    stds_slowdown = []
    results_speedup = []
    stds_speedup = []
    for exp in exps:
        slowdown_mean, slowdown_std = compute_slowdown(exp, 'pc_int_recover')
        #speedup_mean, speedup_std = compute_slowdown(exp, 'pc_int_early')
        #results_speedup.append(speedup_mean)
        #stds_speedup.append(speedup_std)
        results_slowdown.append(slowdown_mean)
        stds_slowdown.append(slowdown_std)

    br1 = np.arange(len(results_slowdown))
    br2 = np.array([x + 0.25 for x in br1])

    fig, ax = plt.subplots(figsize=(12, 10))


    #plt.bar(br1, pc_int_means, width=0.25, color='r', label='Interaction@5', yerr=pc_int_stds)
    #plt.bar(br2, pc_int_early_means, width=0.25, color='b', label='Interaction@10', yerr=pc_int_early_stds)

    ax.bar(br1, results_slowdown, yerr=stds_slowdown, width=0.25, color='#51829B', label='IBO-HPC w/ harm. feedback')
    #ax.bar(br2, results_speedup, yerr=stds_speedup, width=0.25, color='#F6995C', label='IBO-HPC w/ ben. feedback')
    #ax.hlines(1.0, xmax=len(results_slowdown) - .5, xmin=-.5, color='#8D493A', label='IBO-HPC w/o feedback')

    #for k, v in bp1.items():
    #    if k not in ['whiskers', 'fliers', 'caps']:
    #        plt.setp(bp1.get(k), color='#6b7ff2')

    tick_pos = [br1[i] for i in range(len(br1))]
    plt.xticks(tick_pos, ['NAS-101 \n (C-10)', 'NAS-201 \n (C-10)', 'JAHS \n (C-10)', 'JAHS \n (FM)', 'JAHS \n(CH)'], size=25)
    lbls = ax.get_yticklabels()
    lbls = [float(l.get_text()) + 1 for l in lbls]
    lbls[0] = ''
    ax.set_yticklabels(lbls)

    plt.ylabel('Relative RT w.r.t. std. IBO-HPC')
    plt.xlabel('Benchmark')
    #plt.ylim(2, -10)
    plt.grid(axis='y', alpha=0.5)
    #plt.ylim(0.0, 18.)

    #ax.legend([bp1["boxes"][0]], ['Interaction@5'], loc='lower right')
    plt.legend()
    plt.savefig(file, bbox_inches='tight')

def plot_no_catchup_diffs_on_neg_feedback(file):
    plt.rcParams.update({'font.size': 30, 'font.family':'serif'})
    #plt.rc('axes', titlesize=20)
    exps = ['nas101', 'nas201', 'jahs_cifar10', 'jahs_fm', 'jahs_co']
    exps = ['jahs_cifar10', 'jahs_co']

    non_recover_mean_perfs, mean_perfs = [], []
    non_recover_perf_stds, perfs_stds = [], []

    for exp in exps:
        non_recovered_perfs, perfs = compute_no_catch_up_diff(exp)
        non_recover_mean_perfs.append(np.mean(non_recovered_perfs))
        mean_perfs.append(np.mean(perfs))
        non_recover_perf_stds.append(np.std(non_recovered_perfs))
        perfs_stds.append(np.std(perfs))

    br1 = np.arange(len(non_recover_mean_perfs))
    br2 = np.array([x + 0.25 for x in br1])

    fig, ax = plt.subplots(figsize=(12, 10))

    #plt.bar(br1, pc_int_means, width=0.25, color='r', label='Interaction@5', yerr=pc_int_stds)
    #plt.bar(br2, pc_int_early_means, width=0.25, color='b', label='Interaction@10', yerr=pc_int_early_stds)

    ax.bar(br1, non_recover_mean_perfs, yerr=non_recover_perf_stds, width=0.25, color='#51829B', label='neg. feedback, no recover')
    ax.bar(br2, mean_perfs, yerr=perfs_stds, width=0.25, color='#F6995C', label='no feedback')
    #ax.hlines(0.0, xmax=len(results), xmin=0, color='#8D493A')

    #for k, v in bp1.items():
    #    if k not in ['whiskers', 'fliers', 'caps']:
    #        plt.setp(bp1.get(k), color='#6b7ff2')

    tick_pos = [(br1[i] + br2[i]) / 2 for i in range(len(br1))]
    #plt.xticks(tick_pos, ['NAS-101 \n (C-10)', 'NAS-201 \n (C-10)', 'JAHS \n (C-10)', 'JAHS \n (FM)', 'JAHS \n(CH)'], size=25)
    plt.xticks(tick_pos, ['JAHS \n (C-10)', 'JAHS \n(CH)'], size=25)

    plt.ylabel('Test Accuracy (%)')
    plt.xlabel('Benchmark')
    #plt.ylim(2, -10)
    plt.grid(axis='y', alpha=0.5)
    #plt.ylim(0.0, 18.)

    ax.legend(loc='lower right')

    plt.savefig(file, bbox_inches='tight')

def plot_cdf(exp, file, max_reads=50):
    plt.rcParams.update({'font.size': 40, 'font.family':'serif'})
    prefix = f'../../data/logs_v2/{exp}_'
    #algos = [ 'pc', 'pc_int', 'pc_int_early', 'rs', 'pibo', 'hyperband', 'ls', 'smac']
    #algos = [ 'pc', 'pc_int_autorecover', 'pc_int_multi_new', 'pc_int', 'pc_int_early', 'pc_int_dist', 'rs', 'pibo', 'bopro', 'ls', 'smac']
    algos = [ 'pc', 'pc_int_recover', 'pc_int_many', 'pc_int_late', 'pc_int_early', 'pc_int_dist', 'skoptbo', 'pibo', 'bopro', 'ls', 'smac']
    fig, ax = plt.subplots(figsize=(12, 9.2))
    for a in algos:
        dfs = read_csvs(prefix + a, max_reads=max_reads)
        df = pd.concat(dfs)
        #for df in dfs:
        test_perf = df['test_performance'].to_numpy()
        print(f"MAX: {test_perf.max()}")
        regret = BEST_PERFS[exp] - test_perf
        c, line_type = ALGO_COLORS[a]
        label = ALGO_NAME_MAPPING[a]
            #ax.hist(test_perf, cumulative=True, density=True, color=c)
        #sns.kdeplot(data=test_perf, cumulative=True, ax=ax, color=c, label=label)
        sns.ecdfplot(data=test_perf, ax=ax, color=c, label=label, linestyle=line_type, linewidth=3.5)

    #plt.legend(fontsize=18)
    ax.tick_params(length=12., width=3., direction='inout')
    plt.ylim(-0.05, 1.05)
    plt.xlim(10, 100)
    plt.ylabel('CDF')
    plt.xlabel('Test Accuracy (%)')
    plt.xticks(np.arange(0, 100, 10), np.arange(0, 100, 10))
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(file, bbox_inches='tight')

def plot_transfer_regret(exp, file, max_included=500):
    prefix = f'../../data/{exp}_'
    algos = ['pc', 'pc_deactivated']
    fig, ax = plt.subplots(figsize=(12, 9))
    for a in algos:
        dfs = read_csvs(prefix + a, max_reads=max_included)
        task_dfs = {}
        for df in dfs:
            for t in df['task'].unique():
                tdf = df[df['task'] == t]
                if t not in task_dfs:
                    task_dfs[t] = [tdf]
                else:
                    task_dfs[t].append(tdf)
        
        for t, dfs in task_dfs.items():
            perf = np.column_stack([df['test_performance'].to_numpy() for df in dfs])
            avg_perf = np.mean(perf, axis=1)
            best_perfs_key = f'{exp}_{t}'
            avg_regret = BEST_PERFS[best_perfs_key] - (avg_perf / 100)
            std_perf = np.std(perf, axis=1)
            y = []
            for i in range(1, len(avg_regret)):
                y.append(avg_regret[:i].min())
            y = np.array(y)
            x = np.arange(len(avg_regret) - 1)
            plt.plot(x, y, label=f'{a} @ {t}', linewidth=2., linestyle=TRANSFER_ALGO_LINESTYLE[a], color=TRANSFER_TASK_COLOR[exp][t])
    
    ax.tick_params(length=12., width=3., direction='inout')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Iteration')
    plt.legend()
    plt.savefig(file)
    

parser = argparse.ArgumentParser()
parser.add_argument('--out-file', type=str, help='Output file plot is stored to.')
parser.add_argument('--exp', type=str, help='Experiment to plot. This name applies to the log-directory names, not the experiment names from main.py.')
parser.add_argument('--plot', default='regret_vs_cost', help='Plot to produce. regret_vs_cost will plot the average incumbent regret against the search cost, \
                    regret_per_it will plot the average test error per iteration (not incumben!), \
                    interaction_gain will plot the gain an interaction achieves compared to no interaction, \
                    speedups generates a boxplot of the speedups across runs and \
                    cdf plots the cumulative distribution function of the accuracy achieved over time.')
parser.add_argument('--with-std', action='store_true', help='Whether to plot std or not')
parser.add_argument('--plot-all', action='store_true', help='Plot all in one go')
parser.add_argument('--version', default='v1') # only relevant for interaction_gain plots
parser.add_argument('--uncertainty', default='SE', help='Type of uncertainty plotted. SE=Standard Error, STD=Standard Deviation')
parser.add_argument('--num-exps', default=500, type=int, help='Number of experiments to use. Will be randomly sampled from the log-directory given by the --exp option.')
parser.add_argument('--is-transfer', action='store_true', help='Flag whether the experiments are transfer experiments or not. If so, special handling of logs is required')

args = parser.parse_args()

if args.plot == 'regret_vs_cost':
    if args.plot_all:
        exps = ['jahs_cifar10', 'jahs_co', 'jahs_fm', 'nas101', 'nas201']
        base_path = f'./plots/interactive-hpo/tpm/'
        out_files = [base_path + f'{e}.pdf' for e in exps]
        for exp, of in zip(exps, out_files):
            res = collect_results(exp, args.num_exps, args.uncertainty, args.is_transfer)
            plot_regrets(res, exp, of, args.with_std)
    else:
        res = collect_results(args.exp, args.num_exps, args.uncertainty, args.is_transfer)
        plot_regrets(res, args.exp, args.out_file, args.with_std)
elif args.plot == 'regret_per_it':
    plot_avg_regrets_per_iter(args.exp, args.out_file)
elif args.plot == 'interaction_gain':
    plot_interaction_gain(args.out_file, args.version)
elif args.plot == 'speedups':
    plot_speedups(args.out_file)
elif args.plot == 'cdf':
    plot_cdf(args.exp, args.out_file, args.num_exps)
elif args.plot == 'transfer_regret_vs_cost':
    plot_transfer_regret(args.exp, args.out_file, args.num_exps)
elif args.plot == 'slowdown':
    plot_slowdown(args.out_file)
elif args.plot == 'runtime_split':
    plot_cost_vs_optim_cost(args.out_file)
elif args.plot == 'runtime_pc_vs_smac':
    plot_runtime_comparison(args.out_file, None)
elif args.plot == 'non-recover-diff':
    plot_no_catchup_diffs_on_neg_feedback(args.out_file)
else:
    raise ValueError(f'No such plot type: {args.plot}')