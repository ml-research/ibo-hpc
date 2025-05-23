from .experiment import TransferBenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
from copy import deepcopy
import os
import pandas as pd

class NASBench201TransferExperiment(TransferBenchmarkExperiment):

    def __init__(self, optimizer_name, tasks=['cifar10'], seed=0) -> None:
        assert len(tasks) > 0, 'Cannot take empty list of tasks'
        print(f"STARTING TRANSFER NAS201 EXPERIMENT WITH TASKS {tasks}")
        self.benchmark_name = 'nas201'
        self.tasks = tasks
        self._task_iter = iter(tasks)
        self.benchmark_config = {
            'task': tasks[0]
        }
        self._num_history_runs = 10
        benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, 
                                                        self.benchmark_config)

        self._optimizer_name = optimizer_name
        self.optimizer_config = self.get_optimizer_config(benchmark)
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)
        super().__init__(benchmark, optimizer)

    def _init_next_task_pc(self, task_idx):
        curr_task = self.tasks[task_idx - 1]
        # store logs of current task
        self.histories[curr_task] = deepcopy(self.optimizer.history)
        if task_idx >= len(self.tasks):
            return
        next_task = self.tasks[task_idx]
        cfg = {
            'task': next_task
        }
        self.benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, cfg)
        search_space = self.benchmark.search_space
        self.optimizer.set_search_space(search_space)
        
    def run(self):
        if self._optimizer_name == 'pc':
            self._run_pc_exp()
        elif self._optimizer_name == 'bounding_box':
            pass

    def _run_pc_exp(self):
        for task_idx in range(len(self.tasks)):
            config, performance = self.optimizer.optimize()
            processed_config = {}
            for name, idx in config.items():
                param_def = self.benchmark.search_space.search_space_definition[name]
                processed_config[name] = param_def['allowed'][int(idx)]
            config = processed_config
            print((config, performance))

            # set next task
            self._init_next_task_pc(task_idx + 1)

    def _run_synetune(self):
        base_path = './data/'
        dfs = []
        for task in self.tasks:
            history_file = os.path.join(base_path, f'nas201_optunabo_{task}')
            csv_files = list(os.listdir(history_file))
            for file in csv_files[:self._num_history_runs]:
                df = pd.read_csv(file, index_col=0)



    def evaluate_config(self, cfg, budget=None):
        test_cfg = cfg
        for key, val in cfg.items():
            cfg[key] = int(val)
        if self.benchmark is not None:
            res = self.benchmark.query(test_cfg, budget)
            return res
        
    def get_optimizer_config(self, benchmark):
        assert benchmark is not None, 'Something went wrong instantiating the benchmark'
        if self._optimizer_name == 'smac':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 2000,
                'cfg_selector_retries': 0
            }
        elif self._optimizer_name == 'hyperband':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'min_budget': 1,
                'max_budget': 200,
                'n_trials': 500
            }
        elif self._optimizer_name == 'pc_transfer':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 10,
                'samples_per_iter': 20,
                'use_eic': False,
                'eic_samplings': 20,
                'deactivate_transfer': False,
            }
        elif self._optimizer_name == 'rs':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 200
            }
        elif self._optimizer_name == 'ls':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'runs': 150
            }
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')
