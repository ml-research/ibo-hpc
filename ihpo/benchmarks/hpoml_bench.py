from typing import Dict
from hpobench.benchmarks.ml import TabularBenchmark, XGBoostBenchmark
from .benchmark import BaseBenchmark, BenchQueryResult
from ..search_spaces import HPOBenchTabularSearchSpace
import numpy as np


class HPOTabularBenchmark(BaseBenchmark):

    def __init__(self, model, task_id, num_seeds=5) -> None:
        super().__init__()
        self.benchmark = TabularBenchmark(model, task_id)
        self.task_id = task_id
        self.num_seeds = num_seeds

    def query(self, cfg: Dict, budget=None) -> BenchQueryResult:
        scores = self.benchmark.objective_function(
            cfg,
        )
        
        cost = scores['cost']
        score_info = scores['info']
        seeds = list(score_info.keys())
        seeds = np.random.choice(seeds, size=self.num_seeds)
        train_accs = []
        val_accs = []
        test_accs = []
        train_losses, val_losses, test_losses = [], [], []
        for s in seeds:
            train_acc = score_info[s]['train_scores']['acc']
            val_acc = score_info[s]['val_scores']['acc']
            test_acc = score_info[s]['test_scores']['acc']
            train_loss = score_info[s]['train_loss']
            val_loss = score_info[s]['val_loss']
            test_loss = score_info[s]['test_loss']
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)

        res = BenchQueryResult(
            np.mean(train_accs), 
            np.mean(val_accs),
            np.mean(test_accs),
            cost=cost,
            train_loss=np.mean(train_losses),
            val_loss=np.mean(val_losses),
            test_loss=np.mean(test_losses)
            )
        return res
    
    @property
    def search_space(self):
        config_space = self.benchmark.get_configuration_space()
        return HPOBenchTabularSearchSpace(config_space)