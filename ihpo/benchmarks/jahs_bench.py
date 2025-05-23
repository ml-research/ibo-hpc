from typing import Dict
from .benchmark import BaseBenchmark, BenchQueryResult
from jahs_bench.api import Benchmark
from ..search_spaces import JAHSBenchSearchSpace

class JAHSBenchmark(BaseBenchmark):

    def __init__(self, task, save_dir='./benchmark_data/') -> None:
        super().__init__()
        self.benchmark = Benchmark(
            task=task,
            kind='surrogate',
            download=True,
            save_dir=save_dir,
            metrics=['valid-acc', 'train-acc', 'test-acc', 'runtime']
            )
        self._EPOCHS = 200 # highest fidelity
        self._search_space = JAHSBenchSearchSpace(self.benchmark)

    def get_min_and_max(self):
        return 0, 100
        
    def query(self, cfg: Dict, budget=None):
        epochs = self._EPOCHS if budget is None else int(round(budget))
        results = self.benchmark(cfg)
        val_acc = results[self._EPOCHS]['valid-acc']
        train_acc = results[self._EPOCHS]['train-acc']
        test_acc = results[self._EPOCHS]['test-acc']
        rt = results[self._EPOCHS]['runtime']
        return BenchQueryResult(
            train_acc,
            val_acc,
            test_acc,
            cost=rt,
            epochs=epochs,
        )
    
    @property
    def search_space(self):
        return self._search_space