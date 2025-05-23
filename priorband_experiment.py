import logging

import numpy as np
import pandas as pd

import neps

from ihpo.benchmarks import JAHSBenchmark, NAS101Benchmark,  NAS201Benchmark, HPOBBenchmark, LCBenchmark, FCNetBenchmark, BenchQueryResult

import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark')
parser.add_argument('--log-dir')

args = parser.parse_args()

if args.benchmark == 'jahs_cifar10':
    benchmark = JAHSBenchmark(task='cifar10', save_dir='./data/')
elif args.benchmark == 'jahs_fmnist':
    benchmark = JAHSBenchmark(task='fashion_mnist', save_dir='./data/')
elif args.benchmark == 'jahs_co':
    benchmark = JAHSBenchmark(task='colorectal_histology', save_dir='./data/')
elif args.benchmark == 'nas101':
    benchmark = NAS101Benchmark(task='cifar10', save_dir='./data/')
elif args.benchmark == 'nas201':
    benchmark = NAS201Benchmark(task='cifar10', save_dir='./data/')
elif args.benchmark == 'hpob':
    benchmark = HPOBBenchmark('./data/hpob-data/', './data/hpob-surrogates/', '6794', '9914')
elif args.benchmark == 'lc':
    benchmark = LCBenchmark(task='vehicle')
elif args.benchmark == 'fcnet':
    benchmark = FCNetBenchmark('slice_localization')

def save(file, history):
    print("============save experiment===================")
    print(file)
    run_history = history
    res_cols = BenchQueryResult.SUPPORTED_METRICS
    prototype_cfg = run_history[0][0]
    cols = res_cols + list(prototype_cfg.keys())
    df_dict = {c: [] for c in cols}
    # add iteration column
    df_dict['iter'] = []
    for cfg, res, iteration in run_history:
        for k, v in cfg.items():
            df_dict[k].append(v)
        
        for k in res_cols:
            df_dict[k].append(res[k])
        
        df_dict['iter'].append(iteration)
    df = pd.DataFrame.from_dict(df_dict)
    df.to_csv(file)

confidences = {
    'jahs_cifar10': {"N": [3, 'high'], "W": [16, 'high'], "Resolution": [1, 'high']},
    'jahs_fmnist': {"N": [3, 'high'], "W": [16, 'high'], "Resolution": [1, 'high']},
    'jahs_co': {"N": [3, 'high'], "W": [16, 'high'], "Resolution": [1, 'high']},
    'nas201': {"Op_0": [2, 'high'], "Op_1": [2, 'high'], "Op_2": [0, 'high']},
    'nas101': {
            "e_0_1": [1, 'high'], 
            "e_0_2": [0, 'high'], 
            "e_0_3": [1, 'high'], 
            "e_0_4": [0, 'high'], 
            "e_0_5": [1, 'high'], 
            "e_0_6": [1, 'high'], 
            "e_1_2": [1, 'high'],
            "e_1_3": [0, 'high'],
            "e_1_4": [0, 'high'],
            "e_1_5": [0, 'high'], 
            "e_1_6": [0, 'high'],
            "e_2_3": [0, 'high'], 
            "e_2_4": [1, 'high'], 
            "e_2_5": [0, 'high'], 
            "e_2_6": [0, 'high'], 
            "e_3_4": [0, 'high'], 
            "e_3_5": [1, 'high'], 
            "e_3_6": [0, 'high'], 
            "e_4_5": [1, 'high'], 
            "e_4_6": [0, 'high'], 
            "e_5_6": [1, 'high']
        },
    #'hpob': {"eta": [0.5, 'high'], 
    #        "subsample": [0.6, 'high'], 
    #        "lambda": [-75, 'high'], 
    #        "min_child_weight": [-12.5, 'high']},
    'hpob': {
            "num_trees": [1700, 'high'], 
            "mtry": [32, 'high'],
            "min_node_size": [793, 'high']
        },
    "lc": {
        "batch_size": [50, 'medium'],
        "num_layers": [2, 'medium'],
        "learning_rate": {0.085, 'medium'}
    },
    "fcnet": {
        "activation_fn_1": ['relu', 'low'],
        "activation_fn_2": ['tanh', 'low'],
        "n_units_1": [512, 'low'],
        "n_units_2": [512, 'low']
    }
}

fidelities = {
    'jahs_cifar10': (1, 200),
    'jahs_fmnist': (1, 200),
    'jahs_co': (1, 200),
    'nas101': (4, 108),
    'nas201': (1, 199),
    'hpob': (1, 19),
    'lc': (6, 50),
    'fcnet': (1, 99)
}

iters = {
    'jahs_cifar10': 2000,
    'jahs_fmnist': 2000,
    'jahs_co': 2000,
    'nas101': 2000,
    'nas201': 2000,
    'hpob': 100,
    'lc': 100,
    'fcnet': 100
}

run_results = []

def run_pipeline(**config):
    if args.benchmark != 'hpob':
        fidelity = config['fidelity']
        if args.benchmark == 'nas101':
            # select closest valid fidelity
            fidelities = np.array([4, 12, 36, 108])
            diff = fidelities - fidelity
            fid_idx = np.argmin(diff).item()
            fidelity = fidelities[fid_idx]
        elif args.benchmark == 'fcnet':
            fidelities = np.array([6, 12, 25, 50])
            diff = fidelities - fidelity
            fid_idx = np.argmin(diff).item()
            fidelity = fidelities[fid_idx]
        config.pop('fidelity')
    else:
        fidelity = None
    if 'jahs' in args.benchmark:
        config.pop('epoch')
        config['Optimizer'] = 'SGD'
    res = benchmark.query(config, fidelity)
    run_results.append((config, res))
    return -res.val_performance

pipeline_space = {}
for k, v in benchmark.search_space.get_search_space_definition().items():
    confs = confidences[args.benchmark]
    if 'allowed' in v:
        rnd_idx = np.random.randint(0, len(v['allowed']), size=1).item()
        default, conf = v['allowed'][rnd_idx], 'medium'
    else:
        default, conf = np.random.uniform(v['min'], v['max'], size=1).item(), 'medium'
    if v['dtype'] == 'float':
        pipeline_space[k] = neps.Float(lower=v['min'], upper=v['max'], default=default, default_confidence=conf)
    elif v['dtype'] == 'int':
        pipeline_space[k] = neps.Integer(lower=min(v['allowed']), upper=max(v['allowed']), default=default, default_confidence=conf)
    else:
        pipeline_space[k] = neps.Categorical(choices=v['allowed'], default=default, default_confidence=conf)

min_fid, max_fid = fidelities[args.benchmark]
pipeline_space['fidelity'] = neps.Integer(lower=min_fid, upper=max_fid, is_fidelity=True)

logging.basicConfig(level=logging.INFO)

root_dir = f"./data/priorband_{args.benchmark}_c"

for i in range(50):
    np.random.seed(i + 100)
    if os.path.exists(root_dir):
        os.system(f'rm -rf {root_dir}')

    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory=root_dir,
        max_evaluations_total=iters[args.benchmark],  # For an alternate stopping method see multi_fidelity.py
        searcher='priorband'
    )

    run_results_with_iters = []
    for j, (cfg, res) in enumerate(run_results):
        run_results_with_iters.append((cfg, res, j))

    exp_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    file_name = f'{args.benchmark}__{exp_time}.csv'
    file_name = os.path.join(args.log_dir, file_name)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    
    save(file_name, run_results_with_iters)

    run_results = []