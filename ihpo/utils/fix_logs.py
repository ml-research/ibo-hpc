"""
    This script fixes logs of SMAC, BOPrO and PiBO.
    Unfortunately, the implementations of these algorithms stop at some criterion that cannot be set by the user.
    Thus, sometimes these methods stop early since they don't find any better configuration anymore.
    This leads to unreflected variances in later iterations in the plots, hence we extend the logs by the best result
    until all iterations are covered.
"""

import pandas as pd
import numpy as np
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--to-fix', default='nas101')

args = parser.parse_args()

if args.to_fix == 'nas101':

    dirs = ['../../data/logs_v2/nas101_bopro_recover_old/']
    for d in dirs:
        print(f"Handle {d}")
        files = os.listdir(d)
        for f in files:
            print(f)
            file = os.path.join(d, f)
            df = pd.read_csv(file, index_col=0)
            max_perf = df['test_performance'].max()
            max_iter = df['iter'].max()
            if max_iter < 1999:
                row = list([df[df['test_performance'] == max_perf].to_numpy()[0]])
                missing_iters = 2000 - max_iter
                iterations = np.arange(max_iter, 2000)
                r_rows = np.array(row * missing_iters)
                new_df = pd.DataFrame(r_rows, columns=df.columns)
                new_df['iter'] = iterations
                df = pd.concat((df, new_df))
                df.to_csv(file)

elif args.to_fix == 'hpob':
    dirs = [dir_ for dir_ in os.listdir('../data/') if dir_.startswith('hpob_5')]
    for d in dirs:
        files = os.listdir(os.path.join('../data/', d))
        for f in files:
            file = os.path.join('../data', d, f)
            df = pd.read_csv(file, index_col=0)
            for col_name in ['train_performance', 'val_performance', 'test_performance']:
                x = df[col_name]
                y = x.apply(lambda val: val[1:-1] if not isinstance(val, float) else val)
                df[col_name] = y.astype(np.float32)
            df.to_csv(file)
else:
    print("Nothing to fix!")