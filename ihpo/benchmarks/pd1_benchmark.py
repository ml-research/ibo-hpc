from typing import Dict
from .benchmark import BaseBenchmark, BenchQueryResult
from ..search_spaces import PD1SearchSpace
from ..consts import PD1_TASKS
import numpy as np
import os
import pickle

class PD1Benchmark(BaseBenchmark):

    def __init__(self, task, save_dir='./benchmark_data/pd1/surrogates/') -> None:
        super().__init__()
        self._search_space = PD1SearchSpace()
        self._file_path = save_dir
        self.task = task
        self.surrogate_cache = {}
        self._load_surrogates()

    def _load_surrogates(self):
        """
            Load the trained surrogates representing the objective function.
        """
        for task in PD1_TASKS:
            surrogate_task_path = os.path.join(self._file_path, task)
            train_surrogate_file = surrogate_task_path + '/best_train'
            val_surrogate_file = surrogate_task_path + '/best_valid'
            with open(train_surrogate_file, 'rb') as f:
                train_surrogate = pickle.load(f)
                self.surrogate_cache[task + '_train'] = train_surrogate
            with open(val_surrogate_file, 'rb') as f:
                val_surrogate = pickle.load(f)
                self.surrogate_cache[task + '_valid'] = val_surrogate
        

    def query(self, cfg: Dict, budget=None) -> BenchQueryResult:
        keys = ['_valid', '_train']

        surrogate_id = self.task + keys[0]

        benchmark_args = {}
        sorted_cfg = {k: cfg[k] for k in self._search_space.get_search_space_definition().keys()}
        x = np.array(list(sorted_cfg.values())).reshape(1, -1)
        for k in keys:
            surrogate_id = self.task + k
            surrogate = self.surrogate_cache[surrogate_id]
            y = surrogate.predict(x).flatten()[0]
            if k == '_train':
                benchmark_args['train_performance'] = y 
            elif k == '_valid':
                benchmark_args['val_performance'] = y
                benchmark_args['test_performance'] = y
                
        return BenchQueryResult(**benchmark_args)
        
    
    def get_min_and_max(self):
        return [0, 1]
    
    @property
    def search_space(self):
        return self._search_space