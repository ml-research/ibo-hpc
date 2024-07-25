import jahs_bench

from jahs_bench.api import Benchmark
from jahs_bench.lib.core.configspace import joint_config_space

import numpy as np
from icecream import ic

from numpy.random.mtrand import RandomState

from spn.algorithms.LearningWrappers import learn_mspn
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.Sampling import sample_instances
from spn.io.Text import spn_to_str_equation
from spn.algorithms.Statistics import get_structure_stats
from spn.algorithms.Inference import log_likelihood

from datetime import datetime as dt
import os
import csv
import sys
import random

import pdb


def save_on_file(results, optimizer_name='', seed=None, metadata_dir='./results/'):
    seed_str = str(seed) if seed is not None else ''
    timestamp = dt.strftime(dt.now(), '%Y-%m-%d-%H-%M-%S')
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    fields = ['time_spent', 'valid_acc', 'test_acc']

    file_pathname = metadata_dir + '{}_{}_{}.csv'.format(optimizer_name, seed_str, timestamp)
    with open(file_pathname, 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(results)
        f.close()


def run_random_search(seed=None, n_iters=200, dataset="cifar10", transfer_learning=False, model_path=".", n_epochs=200,
                      transfer_dataset="fashion_mnist"):

    # N_ITERATIONS = 4120

    if not seed:
        seed = 123

    np.random.seed(seed)
    random.seed(seed)

    print("loading benchmark class...")

    benchmark = Benchmark(
            task=dataset,
            save_dir=model_path,
            kind="surrogate",
            download=True
        )

    print("done...")

    # Random Search
    configs = []
    results = []
    full_results = []

    print("start search...")

    for it in range(n_iters):
        # Use benchmark ConfigSpace object to sample a random configuration.
        config = joint_config_space.sample_configuration().get_dictionary()
        # Alternatively, define configuration as a dictionary.
        # config = {
        #     'Optimizer': 'SGD',
        #     'LearningRate': 0.1,
        #     'WeightDecay': 5e-05,
        #     'Activation': 'Mish',
        #     'TrivialAugment': False,
        #     'Op1': 4,
        #     'Op2': 1,
        #     'Op3': 2,
        #     'Op4': 0,
        #     'Op5': 2,
        #     'Op6': 1,
        #     'N': 5,
        #     'W': 16,
        #     'Resolution': 1.0,
        # }
        result = benchmark(config, nepochs=n_epochs)

        configs.append(config)
        print(config)
        print(100 - float(result[n_epochs]["valid-acc"]))
        results.append(100 - float(result[n_epochs]["valid-acc"]))
        full_results.append([result[n_epochs]["runtime"], result[n_epochs]["valid-acc"], result[n_epochs]["test-acc"]])

        if transfer_learning and it == 100:
            print("Transfer learning, switching to a different benchmark...")
            benchmark = Benchmark(
                task=transfer_dataset,
                save_dir=model_path,
                kind="surrogate",
                download=True
            )

    incumbent_idx = min(range(len(results)), key=results.__getitem__)
    incumbent = configs[incumbent_idx]
    incumbent_value = results[incumbent_idx]
    print(f"Incumbent: {incumbent} \n Incumbent Value: {incumbent_value}")

    full_results = np.array(full_results)
    save_on_file(full_results, optimizer_name='RandomSearch', seed=seed)
    print("Results saved on file.")

    return seed

    # sys.exit()
    # breakpoint()
    #
    # incumbent['LearningRate'] = 1.0
    # print(incumbent)
    # result = benchmark(incumbent, nepochs=n_epochs)
    # print(100 - float(result[n_epochs]["valid-acc"]))
    #
    # print("*"*80)
    # print("GOOD CONFIG")
    # good_config = {'Activation': 'ReLU', 'LearningRate': 0.06780767038320416, 'N': 5, 'Op1': 3, 'Op2': 2, 'Op3': 3,
    #                'Op4': 0, 'Op5': 0, 'Op6': 2, 'Optimizer': 'SGD', 'Resolution': 1.0, 'TrivialAugment': True,
    #                'W': 16, 'WeightDecay': 0.000539001748734346}
    # print(good_config)
    # result = benchmark(good_config, nepochs=n_epochs)
    # print(100 - float(result[n_epochs]["valid-acc"]))
    #
    # good_config['LearningRate'] = 1.0
    # print(good_config)
    # result = benchmark(good_config, nepochs=n_epochs)
    # print(100 - float(result[n_epochs]["valid-acc"]))
    #
    # good_config['WeightDecay'] = 0.1
    # print(good_config)
    # result = benchmark(good_config, nepochs=n_epochs)
    # print(100 - float(result[n_epochs]["valid-acc"]))


def run():
    benchmark = jahs_bench.Benchmark(task="cifar10", kind="surrogate", download=True)
    config = benchmark.sample_config()
    results = benchmark(config, nepochs=200)
    print(f"Sampled random configuration: {config}\nResults of query on the surrogate "
          f"model: {results}")


def transform_config(config=None):
    assert config

    activation_dict = {'ReLU': 0, 'Hardswish': 1, 'Mish': 2}
    resolution_dict = {0.25: 0, 0.5: 1, 1.0: 2}
    w_dict = {4: 0, 8: 1, 16: 2}
    n_dict = {1: 0, 3: 1, 5: 2}

    values_list = list(config.values())
    values_list[0] = activation_dict[values_list[0]]
    values_list[2] = n_dict[values_list[2]]
    values_list[10] = resolution_dict[values_list[10]]
    values_list[11] = 1 if values_list[11] else 0
    values_list[12] = w_dict[values_list[12]]

    # LR should be in [10e-3, 1]
    # WD 10e-5, 10e-2

    # NOTE delete "Optimizer" field, 9-th element
    del values_list[9]

    return values_list


def pc_sampling(seed=None, n_iters=200, n_init_iters=100, dataset="cifar10", transfer_learning=False, model_path=".",
                n_epochs=200, transfer_dataset="fashion_mnist"):

    if not seed:
        seed = 123

    np.random.seed(seed)
    random.seed(seed)
    random_state = RandomState(seed)

    num_samples = 20

    print("loading benchmark class...")

    benchmark = Benchmark(
        task=dataset,
        save_dir=model_path,
        kind="surrogate",
        download=True
    )
    print("done...")

    # Random Search
    configs = []
    results = []
    pc_data = []
    full_results = []

    print("start search...")

    for it in range(n_init_iters):
        # Use benchmark ConfigSpace object to sample a random configuration.
        config = joint_config_space.sample_configuration().get_dictionary()
        # Alternatively, define configuration as a dictionary.
        # config = {
        #     'Optimizer': 'SGD',
        #     'LearningRate': 0.1,
        #     'WeightDecay': 5e-05,
        #     'Activation': 'Mish',
        #     'TrivialAugment': False,
        #     'Op1': 4,
        #     'Op2': 1,
        #     'Op3': 2,
        #     'Op4': 0,
        #     'Op5': 2,
        #     'Op6': 1,
        #     'N': 5,
        #     'W': 16,
        #     'Resolution': 1.0,
        # }
        result = benchmark(config, nepochs=n_epochs)

        configs.append(config)
        print(config)
        print(100 - float(result[n_epochs]["valid-acc"]))
        results.append(100 - float(result[n_epochs]["valid-acc"]))

        transformed_config = transform_config(config)
        pc_data_row = transformed_config
        # pc_data.append([result[n_epochs]["runtime"], result[n_epochs]["valid-acc"]])
        pc_data_row.append(result[n_epochs]["valid-acc"])
        pc_data.append(pc_data_row)

        full_results.append([result[n_epochs]["runtime"], result[n_epochs]["valid-acc"], result[n_epochs]["test-acc"]])

    print("random init completed...")
    pc_data = np.array(pc_data)

    to_resolution_dict = {0: 0.25, 1: 0.5, 2: 1.0}
    to_activation_dict = {0: 'ReLU', 1: 'Hardswish', 2: 'Mish'}
    to_w_dict = {0: 4, 1: 8, 2: 16}
    to_n_dict = {0: 1, 1: 3, 2: 5}

    performed_evals = 0

    while performed_evals < n_iters:
        # learn a pc
        # TODO double check types, if data needs a conversion (e.g. real -- discrete)
        meta_types_ary = ([MetaType.DISCRETE, MetaType.REAL] + [MetaType.DISCRETE] * 8 +
                          [MetaType.BINARY, MetaType.DISCRETE, MetaType.REAL, MetaType.REAL])

        ds_context = Context(meta_types=meta_types_ary)
        # breakpoint()
        ds_context.add_domains(pc_data)

        # NOTE mu should be adaptive
        mu_hp = max(3, pc_data.shape[0] // 100)
        mspn = learn_mspn(pc_data, ds_context, min_instances_slice=mu_hp, rand_gen=random_state)
        ic(get_structure_stats(mspn))

        # sample configurations
        cond_array = [np.nan] * (pc_data.shape[1])

        # conditioning on the max valid accuracy (observed so far)
        cond_array[-1] = pc_data[:, -1].max()
        samples = sample_instances(mspn, np.array([cond_array] * num_samples), random_state)

        # eval configurations and repeat
        for sample in samples:
            config = {  # NOTE we need to keep the order consistent with transform_config() method
                'Activation': to_activation_dict[sample[0]],
                'LearningRate': max(1e-3, min(sample[1], 1)),
                'N': to_n_dict[sample[2]],
                'Op1': sample[3],
                'Op2': sample[4],
                'Op3': sample[5],
                'Op4': sample[6],
                'Op5': sample[7],
                'Op6': sample[8],
                'Optimizer': 'SGD',
                'Resolution': to_resolution_dict[sample[9]],
                'TrivialAugment': True if sample[10] == 1 else False,
                'W': to_w_dict[sample[11]],
                'WeightDecay': max(1e-5, min(sample[12], 1e-2)),
            }
            print(config)

            result = benchmark(config, nepochs=n_epochs)
            performed_evals += 1

            configs.append(config)
            results.append(100 - float(result[n_epochs]["valid-acc"]))

            # TODO this could be taken directly with the PC 'format'
            transformed_config = transform_config(config)
            pc_data_row = transformed_config
            # pc_data.append([result[n_epochs]["runtime"], result[n_epochs]["valid-acc"]])
            pc_data_row.append(result[n_epochs]["valid-acc"])
            pc_data_row = np.array(pc_data_row)
            pc_data_row = pc_data_row.reshape(1, -1)
            pc_data = np.vstack((pc_data, pc_data_row))
            ic(pc_data.shape)

            full_results.append([result[n_epochs]["runtime"],
                                 result[n_epochs]["valid-acc"],
                                 result[n_epochs]["test-acc"]])

            if transfer_learning and performed_evals == 200:
                print("Transfer learning, switching to a different benchmark...")
                benchmark = Benchmark(
                    task=transfer_dataset,
                    save_dir=model_path,
                    kind="surrogate",
                    download=True
                )

    incumbent_idx = min(range(len(results)), key=results.__getitem__)
    incumbent = configs[incumbent_idx]
    incumbent_value = results[incumbent_idx]
    print(f"Incumbent: {incumbent} \n Incumbent Value: {incumbent_value}")

    full_results = np.array(full_results)
    save_on_file(full_results, optimizer_name='PC_SAMPLING', seed=seed)
    print("Results saved on file.")

    return seed


if __name__ == "__main__":
    print("Starting...")
    # pdb.set_trace()
    # run()
    # run_random_search(seed=55555)
    # pc_sampling()
    # pc_sampling(seed=55555, n_iters=100, n_init_iters=100, dataset="fashion_mnist", transfer_learning=False)
    # pc_sampling(seed=55555, n_iters=300, n_init_iters=100, dataset="cifar10", transfer_learning=True)
    # pc_sampling(seed=55555, n_iters=100, n_init_iters=100, dataset="fashion_mnist", transfer_learning=True)
    # pc_sampling(seed=55555, n_iters=300, n_init_iters=100, dataset="cifar10", transfer_learning=True,
    #             transfer_dataset="colorectal_histology")
    pc_sampling(seed=55555, n_iters=100, n_init_iters=100, dataset="colorectal_histology", transfer_learning=False,
                transfer_dataset="colorectal_histology")
    # pc_sampling()
    print("Quitting...")
    sys.exit()

    # import time
    # random_seeds = [99, 100, 2020, 1234, 5555, 22, 81, 345, 2024, 11]
    # for rs in random_seeds:
    #     run_random_search(seed=rs)
    #     time.sleep(5)

    # for rs in random_seeds:
    #     pc_sampling(seed=rs)
    #     time.sleep(5)

    # sys.exit()
