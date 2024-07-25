import pandas as pd
from ..benchmarks import BenchQueryResult, Benchmark
from ..optimizers import Optimizer

class Experiment:

    def __init__(self) -> None:
        pass

    def run(self):
        raise NotImplementedError('Experiment has no implementation for run')
    
class BenchmarkExperiment(Experiment):

    def __init__(self, benchmark: Benchmark, optimizer: Optimizer) -> None:
        super().__init__()
        self.benchmark = benchmark
        self.optimizer = optimizer

    def save(self, file):
        print("============save experiment===================")
        print(file)
        run_history = self.optimizer.history
        res_cols = BenchQueryResult.SUPPORTED_METRICS
        prototype_cfg = run_history[0][0]
        cols = res_cols + list(prototype_cfg.keys())
        df_dict = {c: [] for c in cols}
        # add iteration column
        df_dict['iter'] = []
        for cfg, res, iteration in run_history:
            for k, v in cfg.items():
                df_dict[k].append(v)
            
            for k in res_cols:
                df_dict[k].append(res[k])
            
            df_dict['iter'].append(iteration)

        df = pd.DataFrame.from_dict(df_dict)
        df.to_csv(file)

class TransferBenchmarkExperiment(Experiment):

    def __init__(self, benchmark: Benchmark, optimizer: Optimizer) -> None:
        super().__init__()
        self.benchmark = benchmark
        self.optimizer = optimizer
        self.histories = {}

    def save(self, file):
        df_dict = {}
        for t, run_history in self.histories.items():
            res_cols = BenchQueryResult.SUPPORTED_METRICS
            for cfg, res, iteration in run_history:
                for k, v in cfg.items():
                    if k not in df_dict:
                        df_dict[k] = []
                    df_dict[k].append(v)
                
                for k in res_cols:
                    if k not in df_dict:
                        df_dict[k] = []
                    df_dict[k].append(res[k])
                
                if 'iter' not in df_dict and 'task' not in df_dict:
                    df_dict['iter'] = []
                    df_dict['task'] = []
                df_dict['iter'].append(iteration)
                df_dict['task'].append(t)

        df = pd.DataFrame.from_dict(df_dict)
        df.to_csv(file)