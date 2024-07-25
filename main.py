from ihpo.experiments import HPOExperiment, NASBench101Experiment, NASBench201Experiment,\
      TransNASBenchExperiment, JAHSBenchExperiment, NASBench101InteractiveExperiment, NASBench201InteractiveExperiment,\
      JAHSBenchInteractiveExperiment, TransNASInteractiveBenchExperiment, NASBench201TransferExperiment
import argparse
import os
from datetime import datetime
from ihpo.utils import setup_environment, create_dir, SEEDS
from rtpt import RTPT
import numpy as np
from multiprocessing import Process

def get_experiment(args):

    if args.exp == 'hpo':
        experiment = HPOExperiment(args.optimizer, args.task)
    elif args.exp == 'nas101':
        experiment = NASBench101Experiment(args.optimizer, args.task)
    elif args.exp == 'nas201':
        experiment = NASBench201Experiment(args.optimizer, args.task)
    elif args.exp == 'transnas':
        experiment = TransNASBenchExperiment(args.optimizer, args.task)
    elif args.exp == 'jahs':
        experiment = JAHSBenchExperiment(args.optimizer, args.task)
    elif args.exp == 'nas101_int':
        experiment = NASBench101InteractiveExperiment(args.optimizer, args.interaction_idx, args.task,)
    elif args.exp == 'nas201_int':
        experiment = NASBench201InteractiveExperiment(args.optimizer, args.interaction_idx, args.task)
    elif args.exp == 'jahs_int':
        experiment = JAHSBenchInteractiveExperiment(args.optimizer, args.interaction_idx, args.task)
    elif args.exp == 'transnas_int':
        experiment = TransNASInteractiveBenchExperiment(args.optimizer, args.interaction_idx, args.task)
    elif args.exp == 'nas201_transfer':
        experiment = NASBench201TransferExperiment(args.optimizer, args.tasks)
    else:
        raise ValueError(f'No such experiment: {args.exp}. Must be hpo, nas101, nas201, transnas or jahs.')
    return experiment


def run_experiment(args, seed_idx):
    exp_counter = 0
    rt = RTPT('JS', 'IHPO', 1)
    rt.start()
    # set random seed for each experiment
    #seed = np.random.choice(np.array(SEEDS))
    seed = SEEDS[seed_idx]
    np.random.seed(seed)
    
    exp_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    file_name = f'{args.exp}_{args.task}_{exp_time}_{seed}.csv'
    log_file = os.path.join(args.log_dir, file_name)
    experiment = get_experiment(args)
    try:
        # PC fails sometimes, ignore these runs
        experiment.run()
        experiment.save(log_file)
        exp_counter += 1
        rt.step()
    except Exception as e:
        #experiment.save(log_file)
        #raise e
        print(f"Experiment failed with: {e}")

parser = argparse.ArgumentParser()

parser.add_argument('--num-experiments', default=500, type=int)
parser.add_argument('--optimizer', default='pc', type=str)
parser.add_argument('--exp', default='hpo', type=str)
parser.add_argument('--task', default='cifar10')
parser.add_argument('--tasks', default=['cifar10'], nargs='+')
parser.add_argument('--log-dir', default='./', type=str)
parser.add_argument('--num-procs', default=5, type=int)
parser.add_argument('--interaction-idx', nargs='+', type=int, default=[-1])
parser.add_argument('--seed-offset', type=int, default=0)

args = parser.parse_args()

setup_environment()
create_dir(args.log_dir)

if args.optimizer == 'pc' or args.optimizer == 'pc_transfer':
    print("PC optimizer not parallelizable. Run in single thread mode.")
    exp_counter = 0
    while exp_counter < args.num_experiments:
        run_experiment(args, args.seed_offset + exp_counter)
else:
    num_procs = min(args.num_procs, args.num_experiments) # min ensures thtat there is at least one batch of processes
    num_batches = int(round(args.num_experiments / num_procs))

    for b in range(num_batches):
        print(f"Start batch {b+1}/{num_batches} of experiments")
        seed_offset = b*num_procs
        processes = [Process(target=run_experiment, args=(args, seed_offset+i)) for i in range(num_procs)]
        for p in processes:
            p.start()

        for p in processes:
            p.join()