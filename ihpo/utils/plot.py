import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os import listdir
from os.path import join
import argparse
from stats import compute_speedups

plt.rcParams.update({'font.size': 20, 'font.family':'serif'})

BEST_PERFS = {
        'nas101': 1, # 0.9452,
        'nas201': 1, #0.945,
        'jahs_cifar10': 1.0,
        'jahs_co': 1.0,
        'jahs_fm': 1.0,
        'nas201_trans_cifar10': 0.93,
        'nas201_trans_cifar100': 0.74
    }

BENCHMARK_LABELS = {
        'nas101': 'NAS-Bench-101',
        'nas201': 'NAS-Bench-201',
        'jahs_cifar10': 'JAHS (CIFAR-10)',
        'jahs_co': 'JAHS (CO)',
        'jahs_fm': 'JAHS (Fashion-MNIST)',
    }

#ALGO_COLORS = {
#        'pc': ('#3492eb', 'solid'),
#        'pc_int': ('#34ebe5', 'solid'),
#        'pc_int_recover': ('#34ebe5', 'solid'),
#        'pc_int_multi': ('#8a4000', 'solid'),
#        'pc_int_early': ('#8a4000', 'solid'),
#        'pibo': ('#014d08', 'dashed'),
#        'smac': ('#ed6618', 'dashed'),
#        'rs': ('#964aed', 'dashed'),
#        'ls': ('#cf06aa', 'dashed'),
#        'hyperband': ('#d10808', 'dashed'),
#    }

ALGO_COLORS = {
        'pc': ('#3492eb', 'solid'),
        'pc_int': ('#34ebe5', 'solid'),
        'pc_int_recover': ('#ea5cfa', 'solid'),
        'pc_int_multi': ('#8a4000', 'solid'),
        'pc_int_early': ('#fa675c', 'solid'),
        'pc_int_dist': ('#15a123', 'solid'),
        'pibo': ('#014d08', 'dotted'),
        'smac': ('#ed6618', 'dashed'),
        'rs': ('#964aed', 'dashed'),
        'ls': ('#cf06aa', 'dashed'),
        #'hyperband': ('#d10808', 'dashed'),
        'bopro': ('#d10808', 'dotted'),
    }


ALGO_NAME_MAPPING = {
        'pc': 'IBO',
        'pc_int': 'IBO (w/ interaction@10)',
        'pc_int_early': 'IBO (w/ interaction@5)',
        'pc_int_multi': 'IBO (w/ interaction@5,20)',
        'pc_int_recover': 'IBO (w/ interaction@5,20)',
        'pc_int_dist': 'IBO (w/ dist. intervention@5)',
        'pibo': 'πBO',
        'smac': 'SMAC',
        'rs': 'RS',
        'rs_test': 'RSL',
        'ls': 'LS',
        'hyperband': 'Hyperband',
        'smac_test': 'SMAC',
        'hyperband_test': 'Hyperband',
        'bopro': 'BOPrO'
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

def compute_regrets(result_path, best_perf, align_cost, num_exps=500, uncertainty='SE'):
    dfs = read_csvs(result_path, num_exps)
    test_perfs = []
    costs = []
    max_iters = int(1e9)

    for df in dfs:
        #iterations = pd.unique(df['iter'])
        max_iters = min(max_iters, df.index.max())
        test_perf_per_iter = []
        cum_cost = []
        for i in range(1, max_iters - 1):
            #iterdf = df[df['iter'] <= i]
            iterdf = df.head(i)
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

def collect_results(exp='nas101', num_exps=500, uncertainty='SE'):
    prefix = f'../../data/logs/{exp}_'
    algos = [ 'pc', 'pc_int', 'pc_int_early', 'pc_int_dist', 'rs', 'pibo', 'bopro', 'ls', 'smac']
    #algos = [ 'pc', 'pc_int_recover', 'pc_int_multi', 'rs', 'pibo', 'hyperband', 'ls', 'smac']
    #algos = ['pc', 'bopro']
    result_paths = [prefix + algo for algo in algos]
    regrets = [(ALGO_NAME_MAPPING[algo], algo, compute_regrets(rp, BEST_PERFS[exp], algo in ['pc', 'pc_int', 'pc_int_early', 'ls'], num_exps, uncertainty)) for algo, rp in zip(algos, result_paths)]
    return regrets

def plot_regrets(regret_results, exp, file='./out.pdf', with_std=False):
    plt.rcParams.update({'font.size': 30, 'font.family':'serif'})
    fig = plt.figure(figsize=(8, 7))
    scales = {
        'jahs_fm': ((0, 0.9e8), (0.048, 0.115)),
        'nas101': ((0, 1.5e6), (0.053, 0.07)),
        'nas201': ((0, 13000), (0.08, 0.125)),
        'jahs_cifar10': ((0, 0.9e8), (0.05, 0.3)),
        'jahs_co': ((0, 0.6e7), (0.05, 0.17))
    }
    for i, (algo, algo_id, res) in enumerate(regret_results):
        mean_perf, perf_std, mean_cost = res[:, 0], res[:, 1], res[:, 2]
        alpha = 1.0 if 'IBO' in algo else 0.4
        c, line_type = ALGO_COLORS[algo_id]
        plt.plot(mean_cost, mean_perf, label=algo, alpha=alpha, linewidth=3., c=c, linestyle=line_type)
        if with_std:
            plt.fill_between(mean_cost, mean_perf-perf_std, mean_perf+perf_std, alpha=0.1, color=c)
    plt.xlabel('wall-clock time (sec.)')
    plt.ylabel('test regret')
    #plt.yscale('log')
    #plt.xscale('log')
    xlim, ylim = scales[exp]
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    #plt.legend()
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
    plt.ylabel('Test regret')

    plt.legend()
    plt.grid()
    plt.savefig(file)


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

    fig, ax = plt.subplots(figsize=(12, 10))

    #plt.bar(br1, pc_int_means, width=0.25, color='r', label='Interaction@5', yerr=pc_int_stds)
    #plt.bar(br2, pc_int_early_means, width=0.25, color='b', label='Interaction@10', yerr=pc_int_early_stds)

    bp1 = ax.boxplot(pc_int, positions=br1, widths=0.25, showfliers=False, boxprops={'linewidth': 4})
    bp2 = ax.boxplot(pc_int_early, positions=br2, widths=0.25, showfliers=False, boxprops={'linewidth': 4})

    for k, v in bp1.items():
        if k not in ['whiskers', 'fliers', 'caps']:
            plt.setp(bp1.get(k), color='#6b7ff2')

    for k, v in bp2.items():
        if k not in ['whiskers', 'fliers', 'caps']:
            plt.setp(bp2.get(k), color='#f26b6b')
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

def plot_cdf(exp, file, max_reads=50):
    plt.rcParams.update({'font.size': 40, 'font.family':'serif'})
    prefix = f'../../data/logs/{exp}_'
    #algos = [ 'pc', 'pc_int', 'pc_int_early', 'rs', 'pibo', 'hyperband', 'ls', 'smac']
    algos = [ 'pc', 'pc_int_recover', 'pc_int_multi', 'pc_int', 'pc_int_early', 'pc_int_dist', 'rs', 'pibo', 'hyperband', 'ls', 'smac']
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
        sns.ecdfplot(data=test_perf, ax=ax, color=c, label=label, linestyle=line_type, linewidth=2.5)

    #plt.legend()
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
                    regret_per_it will plot the average test regret per iteration (not incumben!), \
                    interaction_gain will plot the gain an interaction achieves compared to no interaction, \
                    speedups generates a boxplot of the speedups across runs and \
                    cdf plots the cumulative distribution function of the accuracy achieved over time.')
parser.add_argument('--with-std', action='store_true', help='Whether to plot std or not')
parser.add_argument('--version', default='v1') # only relevant for interaction_gain plots
parser.add_argument('--uncertainty', default='SE', help='Type of uncertainty plotted. SE=Standard Error, STD=Standard Deviation')
parser.add_argument('--num-exps', default=500, type=int, help='Number of experiments to use. Will be randomly sampled from the log-directory given by the --exp option.')

args = parser.parse_args()

if args.plot == 'regret_vs_cost':
    res = collect_results(args.exp, args.num_exps, args.uncertainty)
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
else:
    raise ValueError(f'No such plot type: {args.plot}')