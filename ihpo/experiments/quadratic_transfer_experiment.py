from .experiment import TransferBenchmarkExperiment
from ..optimizers import OptimizerFactory
from ..search_spaces import RealSearchSpace
from ..benchmarks import BenchQueryResult
import numpy as np
import os

class QuadraticTransferExperiment(TransferBenchmarkExperiment):

    def __init__(self, optimizer_name, prior_tasks, num_prior_runs_per_task=5, prior_task_log='./data/', seed=0) -> None:
        benchmark = None
        self._seed = seed
        self._optimizer_name = optimizer_name   
        self.constants = [[1, 1], [2., 2.]]
        self.prior_tasks = prior_tasks
        self._num_prior_runs_per_task = num_prior_runs_per_task
        self._prior_task_log = prior_task_log
        self.search_space = RealSearchSpace(len(self.constants[-1]))
        self.optimizer_config = self.get_optimizer_config()
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)
        self.task_idx = 0
        super().__init__(benchmark, optimizer, prior_task_log)

    def run(self):
        if self._optimizer_name in ['pc_transfer', 'pc_transfer_rf', 'einet']:
            config, performance = self.continual_optimization()
        # perform optimization on second search space
        else:
            config, performance = self.optimizer.optimize()
            self.histories[self.task_idx] = self.optimizer.history
        print(config, performance)

    def continual_optimization(self):
        """
            Start warm-up phase of optimizer. 
            This is only necessary for HyTraLVIP since it works sequentially.
            Other methods simply take log of former tasks.
        """
        mins, maxs = [], []
        for task_idx, constant_vec in enumerate(self.constants):
            evaluate_fun = self.create_evaluate_config(constant_vec)
            self.optimizer.objective = evaluate_fun
            search_space = RealSearchSpace(len(constant_vec))
            self.optimizer.set_search_space(search_space)
            config, performance = self.optimizer.optimize()
            self.histories[task_idx] = self.optimizer.history
            self.task_idx = task_idx
        return config, performance

    def create_evaluate_config(self, consts):
        def evaluate_config(cfg):
            config_vec = list(cfg.values())
            return quadartic(config_vec, consts)
        return evaluate_config

    def get_optimizer_config(self):
        if self._optimizer_name == 'smac':
            return {
                'objective': self.create_evaluate_config(self.constants[-1]),
                'search_space': self.search_space,
                'n_trials': 2000,
                'seed': self._seed
            }
        elif self._optimizer_name == 'einet':
            self.search_space = RealSearchSpace(len(self.constants[0]))
            return {
                'objective': self.create_evaluate_config(self.constants[0]),
                'search_space': self.search_space,
                'iterations': 20,
                'use_ei': True,
            }
        elif self._optimizer_name == 'pc_transfer':
            # set first search space as HyTraLVIP is a sequential method
            self.search_space = RealSearchSpace(len(self.constants[0]))
            return {
                'objective': self.create_evaluate_config(self.constants[0]),
                'search_space': self.search_space,
                'iterations': 20,
                'num_samples': 5,
                'use_ei': False,
                'num_self_consistency_samplings': 200,
                'num_ei_repeats': 100,
                'pc_type': 'parametric',
                'seed': self._seed,
                'transfer_deacy': 0.5,
                'decay_factor': 0.85
            }
        elif self._optimizer_name == 'optunabo':
            return {
                'objective': self.create_evaluate_config(self.constants[-1]),
                'search_space': self.search_space,
                'iterations': 2000
            }
        elif self._optimizer_name == 'skoptbo':
            return {
                'objective': self.create_evaluate_config(self.constants[-1]),
                'search_space': self.search_space,
                'iterations': 1000,
                'surrogate': 'rf'
            },
        elif self._optimizer_name == 'rgpe':
            return {
                'objective': self.create_evaluate_config(self.constants[-1]),
                'search_space': self.search_space,
                'max_iter': 100,
                'method': 'tlbo_rgpe_prf',
                'transfer_learning_evaluation_files': self.get_prior_runs()
            }
        elif self._optimizer_name == 'mphd':
            return {
                'objective': self.create_evaluate_config(self.constants[-1]),
                'search_space': self.search_space,
                'transfer_learning_evaluation_files': self.get_prior_runs(),
                'y_transform': lambda x: x/100,
                'iters': 100,
                'gpu': -1
            }
        elif self._optimizer_name == 'fsbo':
            return {
                'objective': self.create_evaluate_config(self.constants[-1]),
                'search_space': self.search_space,
                'transfer_learning_evaluation_files': self.get_prior_runs(),
                'iters': 200,
                'gpu': -1 
            }
        elif self._optimizer_name == 'ablr':
            return {
                'objective': self.create_evaluate_config(self.constants[-1]),
                'search_space': self.search_space,
                'transfer_learning_evaluation_files': self.get_prior_runs(),
                'iters': 10,
                'gpu': -1,
                'seed': self._seed,
                'num_start_samples': 5, 
                'num_ei_samples': 1000,
                'pretrain_epochs': 100, 
                'fine_tune_epochs': 3,
            }
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')
        
    def get_prior_runs(self):
        return self.load_prior_hpo_logs()
    
    def load_prior_hpo_logs(self):
        task_files = {}

        for task in self.prior_tasks:
            task_name = f'{task}_optunabo' # we use optuna runs as data source as its a good baseline and widely used
            path = os.path.join(self._prior_task_log, f'{task_name}/')
            files = list(os.listdir(path))
            fidx = np.random.randint(0, len(files), size=self._num_prior_runs_per_task)
            selected_files = [path + files[int(idx)] for idx in fidx]
            task_files[task_name] = selected_files

        return task_files


def quadartic(x, c):
    result = -np.sum((np.array(x) + np.array(c))**2)
    return BenchQueryResult(result, result, result, cost=1)