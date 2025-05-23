from typing import Dict
from .benchmark import BaseBenchmark, BenchQueryResult
from naslib.search_spaces import TransBench101SearchSpaceMicro
from naslib.search_spaces.core.query_metrics import Metric
from naslib.utils import get_dataset_api
from ..search_spaces import TransNASSearchSpace

class TransNASBench(BaseBenchmark):

    def __init__(self, task, save_dir='./data/') -> None:
        super().__init__()
        self.benchmark = TransBench101SearchSpaceMicro(dataset=task)
        self.task = task
        self.save_dir = save_dir
        self.dataset_api = get_dataset_api(self.benchmark.space_name, task)
        # get best validation score for scaling
        self.best_val_acc, _ = self.dataset_api['api'].get_best_archs(task, 'valid_top1', 'micro')[0]
        self._search_sapce = TransNASSearchSpace(self.benchmark, self.dataset_api)
        self._EPOCHS = 10 if task=='jigsaw' else 25

    def get_min_and_max(self):
        return 0, self.best_val_acc
        
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
            train_acc,
            val_acc,
            test_acc,
            cost=train_time,
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=test_loss,
            epochs=self._EPOCHS
        )
    
    @property
    def search_space(self):
        return self._search_sapce
