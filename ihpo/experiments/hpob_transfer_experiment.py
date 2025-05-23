
from .experiment import TransferBenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
import os
import numpy as np

class HPOBTransferExperiment(TransferBenchmarkExperiment):

    def __init__(self, optimizer_name, prior_tasks, target_search_space_id, 
                 target_dataset_id, seed=0,  num_pior_runs_per_task=1, 
                 prior_task_log='./data/', impute_missing_hps=False) -> None:
        self.benchmark_name = 'hpob'
        self.search_space_id = target_search_space_id
        self.dataset_id = target_dataset_id
        self.prior_tasks = prior_tasks
        self._seed = seed
        self._impute_missing_hps = impute_missing_hps
        benchmark_cfg = {
            'search_space_id': self.search_space_id,
            'dataset_id': self.dataset_id,
            'surrogate_path': './benchmark_data/hpob-surrogates/',
            'dataset_path': './benchmark_data/hpob-data/'
        }
        self._num_prior_runs_per_task = num_pior_runs_per_task
        self._prior_task_log = prior_task_log
        benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name,
                                                        benchmark_cfg)

        self._optimizer_name = optimizer_name
        self.optimizer_config = self.get_optimizer_config(benchmark)
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)
        super().__init__(benchmark, optimizer, prior_task_log)

    def run(self):
        if self._optimizer_name in ['pc_transfer', 'pc_transfer_rf']:
            config, performance = self.continual_optimization()
        # perform optimization on second search space
        else:
            config, performance = self.optimizer.optimize()
            self.histories[f'{self.search_space_id}_{self.dataset_id}'] = self.optimizer.history
        print(config, performance)
    
    def continual_optimization(self):
        """
            Start warm-up phase of optimizer. 
            This is only necessary for HyTraLVIP since it works sequentially.
            Other methods simply take log of former tasks.
        """
        tasks = self.prior_tasks + [(self.search_space_id, self.dataset_id)]
        for ssid, dsid in tasks:
            benchmark_cfg = {
                'search_space_id': ssid,
                'dataset_id': dsid,
                'surrogate_path': './data/hpob-surrogates/',
                'dataset_path': './data/hpob-data/'
            }
            benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, 
                                                    benchmark_cfg)
            evaluate_fun = self.create_evaluate_config(benchmark)
            self.optimizer.objective = evaluate_fun
            self.optimizer.set_search_space(benchmark.search_space)
            config, performance = self.optimizer.optimize()
            self.histories[f'{ssid}_{dsid}'] = self.optimizer.history
        min_, max_ = benchmark.get_min_and_max()
        performance.inv_scale(min_, max_)
        return config, performance

    def create_evaluate_config(self, benchmark):

        min_, max_ = benchmark.get_min_and_max()

        def evaluate_config(cfg, budget=None):
            test_cfg = cfg
            if benchmark is not None:
                res = benchmark.query(test_cfg, budget)
                res.scale(min_, max_)
                return res
        
        return evaluate_config
        
    def get_optimizer_config(self, benchmark):
        assert benchmark is not None, 'Something went wrong instantiating the benchmark'
        if self._optimizer_name == 'smac':
            return {
                'objective': self.create_evaluate_config(benchmark),
                'search_space': benchmark.search_space,
                'n_trials': 100,
                'num_iter': 2000
            }
        elif self._optimizer_name == 'quant':
            return {
                'objective': self.create_evaluate_config(benchmark),
                'search_space': benchmark.search_space,
                'transfer_learning_evaluation_files': self.get_prior_runs(),
                'max_trials': 2000,
            }
        elif self._optimizer_name == 'bbox':
            return {
                'objective': self.create_evaluate_config(benchmark),
                'search_space': benchmark.search_space,
                'transfer_learning_evaluation_files': self.get_prior_runs(),
                'max_trials': 2000,
            }
        elif self._optimizer_name == 'pc_transfer': 
            return {
                'objective': self.create_evaluate_config(benchmark),
                'search_space': benchmark.search_space,
                'iterations': 20,
                'num_samples': 20,
                'use_ei': False,
                'num_self_consistency_samplings': 100,
                'num_ei_repeats': 20,
                'pc_type': 'parametric',
                'seed': self._seed
            }
        elif self._optimizer_name == 'rs':
            return {
                'objective': self.create_evaluate_config(benchmark),
                'search_space': benchmark.search_space,
                'iterations': 2000,
            }
        elif self._optimizer_name == 'ls':
            return {
                'objective': self.create_evaluate_config(benchmark),
                'search_space': benchmark.search_space,
                'runs': 150
            }
        elif self._optimizer_name == 'transbo':
            return {
                'objective': self.create_evaluate_config(benchmark),
                'search_space': benchmark.search_space,
                'max_iter': 200,
                'method': 'tlbo_topov3_prf',
                'transfer_learning_evaluation_files': self.get_prior_runs()
            }
        elif self._optimizer_name == 'rgpe':
            return {
                'objective': self.create_evaluate_config(benchmark),
                'search_space': benchmark.search_space,
                'max_iter': 200,
                'method': 'tlbo_rgpe_prf',
                'transfer_learning_evaluation_files': self.get_prior_runs()
            }
        elif self._optimizer_name == '0shot':
            return {
                'objective': self.create_evaluate_config(benchmark),
                'search_space': benchmark.search_space,
                'max_trials': 2000,
                'transfer_learning_evaluation_files': self.get_prior_runs()
            }
        elif self._optimizer_name == 'mphd':
            return {
                'objective': self.create_evaluate_config(benchmark),
                'search_space': benchmark.search_space,
                'transfer_learning_evaluation_files': self.get_prior_runs(),
                'y_transform': lambda x: x/100,
                'iters': 100,
                'gpu': -1
            }
        elif self._optimizer_name == 'fsbo':
            return {
                'objective': self.create_evaluate_config(benchmark),
                'search_space': benchmark.search_space,
                'transfer_learning_evaluation_files': self.get_prior_runs(),
                'iters': 200,
                'gpu': -1 
            }
        elif self._optimizer_name == 'pc_transfer_rf': 
            return {
                'objective': self.create_evaluate_config(benchmark),
                'search_space': benchmark.search_space,
                'iterations': 2000,
                'warm_start_samples': 1e4,
                'seed': self._seed
            }
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')
        
    def get_prior_runs(self):
        if self._impute_missing_hps:
            return self.load_imputed_prior_hpo_runs()
        else:
            return self.load_prior_hpo_logs()
        
    def load_prior_hpo_logs(self):
        task_files = {}

        for task in self.prior_tasks:
            task_name = f'{task[0]}_{task[1]}_optunabo' # we use optuna runs as data source as its a good baseline and widely used
            path = os.path.join(self._prior_task_log, f'{self.benchmark_name}_{task_name}/')
            files = list(os.listdir(path))
            fidx = np.random.randint(0, len(files), size=self._num_prior_runs_per_task)
            selected_files = [path + files[int(idx)] for idx in fidx]
            task_files[task_name] = selected_files

        return task_files