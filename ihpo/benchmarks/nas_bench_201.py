from typing import Dict
from .benchmark import BaseBenchmark, BenchQueryResult
from naslib.search_spaces import NasBench201SearchSpace
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils import get_dataset_api
import networkx as nx
from ..search_spaces import NAS201SearchSpace

class NAS201Benchmark(BaseBenchmark):

    def __init__(self, task, save_dir='./benchmark_data/') -> None:
        super().__init__()
        self.benchmark = NasBench201SearchSpace()
        self.task = task
        self.save_dir = save_dir
        self.dataset_api = get_dataset_api(self.benchmark.space_name, task)
        self._search_space = NAS201SearchSpace(self.benchmark, self.dataset_api)
        
    def query(self, cfg: Dict, budget=None):
        epochs = -1 if budget is None else int(round(budget))
        # assumes dict in form {op_i: idx}
        ops = list(cfg.values())
        new_bench = self.benchmark.clone()
        new_bench.set_spec(ops)
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
        train_loss = new_bench.query(Metric.TRAIN_LOSS, self.task, 
                                        self.save_dir, dataset_api=self.dataset_api,
                                        epoch=epochs)
        val_loss = new_bench.query(Metric.VAL_LOSS, self.task, 
                                        self.save_dir, dataset_api=self.dataset_api,
                                        epoch=epochs)
        test_loss = new_bench.query(Metric.TEST_LOSS, self.task, 
                                        self.save_dir, dataset_api=self.dataset_api,
                                        epoch=epochs)
        return BenchQueryResult(
            round(train_acc, 2),
            round(val_acc, 2),
            round(test_acc, 2),
            cost=train_time,
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=test_loss,
            epochs=200
        )
    
    @property
    def search_space(self):
        return self._search_space
