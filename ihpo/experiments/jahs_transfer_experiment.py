from .experiment import TransferBenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory
from ..consts import JAHS_TASKS
import os
import numpy as np

# TODO: Use TransferBenchmarkExperiment base class for more detailed logs!
class JAHSBenchTransferExperiment(TransferBenchmarkExperiment):

    def __init__(self, optimizer_name, prior_tasks, target_task, seed=0, 
                 num_pior_runs_per_task=1, prior_task_log='./data/',
                 is_heterogeneous=False) -> None:
        self.benchmark_name = 'jahs'
        self.prior_tasks = prior_tasks
        if len(self.prior_tasks) == 1 and self.prior_tasks[0] == '*':
            self.prior_tasks = [t for t in JAHS_TASKS if t != target_task]
        self.target_task = target_task
        self._num_prior_runs_per_task = num_pior_runs_per_task
        self._prior_task_log = prior_task_log
        self._seed = seed
        self._is_heterogeneous = is_heterogeneous
        self.benchmark_target_task = {
            'task': target_task
        }
        benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, 
                                                        self.benchmark_target_task)

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
            self.histories[self.target_task] = self.optimizer.history
        print((config, performance))

    def continual_optimization(self):
        """
            Perform continual HPO. Only supported by HyTraLVIP
        """
        tasks = self.prior_tasks + [self.target_task]
        for pt in tasks:
            benchmark_cfg = {
                'task': pt
            }
            benchmark = BenchmarkFactory.get_benchmark(self.benchmark_name, 
                                                    benchmark_cfg)
            evaluate_fun = self.create_evaluate_config(benchmark)
            self.optimizer.objective = evaluate_fun
            self.optimizer.set_search_space(benchmark.search_space)
            config, performance = self.optimizer.optimize()
            self.histories[pt] = self.optimizer.history

        return config, performance

    def create_evaluate_config(self, benchmark):

        def evaluate_config(cfg, budget=None):
            if benchmark is not None:
                # fix fidelities
                cfg['Optimizer'] = 'SGD'
                res = benchmark.query(cfg, budget)
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
                'num_hyperparameters_per_task': 10,
            }
        elif self._optimizer_name == 'pc_transfer': 
            return {
                'objective': self.create_evaluate_config(benchmark),
                'search_space': benchmark.search_space,
                'iterations': 10,
                'num_samples': 5,
                'initial_samples': 10,
                'num_self_consistency_samplings': 50,
                'max_rel_ball_size': 0,
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
                'max_iter': 500,
                'method': 'tlbo_topov3_prf',
                'transfer_learning_evaluation_files': self.get_prior_runs()
            }
        elif self._optimizer_name == 'rgpe':
            return {
                'objective': self.create_evaluate_config(benchmark),
                'search_space': benchmark.search_space,
                'max_iter': 500,
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
                'iters': 100,
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
        if self._is_heterogeneous:
            return self.load_imputed_prior_hpo_runs()
        else:
            return self.load_prior_hpo_logs()
        
    def load_prior_hpo_logs(self):
        task_files = {}

        for task in self.prior_tasks:
            path = os.path.join(self._prior_task_log, f'htl-prior-{self.benchmark_name}-{task}/')
            files = list(os.listdir(path))
            fidx = np.random.randint(0, len(files), size=self._num_prior_runs_per_task)
            selected_files = [path + files[int(idx)] for idx in fidx]
            task_files[task] = selected_files

        return task_files