from typing import Dict
from .benchmark import BaseBenchmark, BenchQueryResult
from ..search_spaces import LCSearchSpace
from ..consts import LC_TASKS
import pandas as pd
import numpy as np
import os
import pickle

class LCBenchmark(BaseBenchmark):

    def __init__(self, task, save_dir='./benchmark_data/lc_bench/') -> None:
        super().__init__()
        self._search_space = LCSearchSpace()
        self._file_path = save_dir
        self.task = task
        self.surrogate_cache = {}
        self.BUDGETS = [6, 12, 25, 50]
        self._load_surrogates()

    def _load_surrogates(self):
        """
            Load the trained surrogates representing the objective function.
        """
        for task in LC_TASKS:
            surrogate_task_path = os.path.join(self._file_path, task)
            for b in self.BUDGETS:
                train_surrogate_file = surrogate_task_path + f'/model_train_at_{b}'
                val_surrogate_file = surrogate_task_path + f'/model_val_at_{b}'
                test_surrogate_file = surrogate_task_path + f'/model_test_at_{b}'
                with open(train_surrogate_file, 'rb') as f:
                    train_surrogate = pickle.load(f)
                    self.surrogate_cache[task + f'_train_{b}'] = train_surrogate
                with open(val_surrogate_file, 'rb') as f:
                    val_surrogate = pickle.load(f)
                    self.surrogate_cache[task + f'_val_{b}'] = val_surrogate
                with open(test_surrogate_file, 'rb') as f:
                    test_surrogate = pickle.load(f)
                    self.surrogate_cache[task + f'_test_{b}'] = test_surrogate

    def query(self, cfg: Dict, budget=None) -> BenchQueryResult:
        keys = ['_val', '_train', '_test']

        if budget is not None and budget in self.BUDGETS:
            epoch = budget 
        else:
            epoch = max(self.BUDGETS)
        benchmark_args = {}
        sorted_cfg = {k: cfg[k] for k in self._search_space.get_search_space_definition().keys()}
        x = np.array(list(sorted_cfg.values())).reshape(1, -1)
        for k in keys:
            surrogate_id = self.task + k + f'_{epoch}'
            surrogate = self.surrogate_cache[surrogate_id]
            y = surrogate.predict(x).flatten()[0]
            if k == '_train':
                benchmark_args['train_performance'] = y 
            elif k == '_val':
                benchmark_args['val_performance'] = y 
            else:
                benchmark_args['test_performance'] = y 
                
        return BenchQueryResult(**benchmark_args)
        
    
    def get_min_and_max(self):
        [0, 1]
    
    @property
    def search_space(self):
        return self._search_space