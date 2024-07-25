from typing import Dict
from .benchmark import BaseBenchmark, BenchQueryResult
from naslib.search_spaces import NasBench101SearchSpace
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils import get_dataset_api
from ..search_spaces import NAS101SearchSpace
import numpy as np
import networkx as nx

class NAS101Benchmark(BaseBenchmark):

    def __init__(self, task, save_dir='./data/') -> None:
        super().__init__()
        self.benchmark = NasBench101SearchSpace()
        self.task = task
        self.save_dir = save_dir
        self.dataset_api = get_dataset_api(self.benchmark.space_name, task)
        self._search_space = NAS101SearchSpace(self.benchmark, self.dataset_api)
        
    def query(self, cfg: Dict, budget=None):
        epochs = -1 if budget is None else int(round(budget))
        spec = self._create_spec(cfg)
        new_bench = self.benchmark.clone()
        try:
            new_bench.set_spec(spec)
            train_time = new_bench.query(Metric.TRAIN_TIME, self.task, 
                                            self.save_dir, dataset_api=self.dataset_api,
                                            epoch=epochs)
            train_acc = new_bench.query(Metric.TRAIN_ACCURACY, self.task, 
                                            self.save_dir, dataset_api=self.dataset_api,
                                            epoch=epochs)
            val_acc = new_bench.query(Metric.VAL_ACCURACY, self.task, 
                                        self.save_dir, dataset_api=self.dataset_api,
                                        epoch=epochs)
            test_acc = new_bench.query(Metric.TEST_ACCURACY, self.task,
                                            self.save_dir, dataset_api=self.dataset_api,
                                            epoch=epochs)
        except Exception as e:
            # some algorithms suggest DAGs not included in benchmark, this will trigger
            # an error. Catch it and set accuracies to -1.
            # TODO: Needed?
            train_time = np.inf
            train_acc = -1
            val_acc = -1
            test_acc = -1
        return BenchQueryResult(
            train_acc,
            val_acc,
            test_acc,
            cost=train_time,
            epochs=108
        )
    
    def _create_spec(self, cfg: Dict):
        adj = np.zeros((7, 7))
        ops = []
        for key, val in cfg.items():
            if key.startswith('e_'):
                i, j = key.split('_')[1:]
                adj[int(i), int(j)] = val
            elif key.startswith('o_'):
                i = key.split('_')[-1]
                ops.append(val)
        adj = adj.astype(np.uint8)
        operations = ['input'] + ops + ['output']
        return {'matrix': adj, 'ops': operations}

    @property
    def search_space(self):
        return self._search_space