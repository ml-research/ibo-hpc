import pandas as pd
from ..benchmarks import BenchQueryResult, Benchmark
from ..optimizers import Optimizer
from ..consts import NO_HYPERPARAMETERS

class Experiment:

    def __init__(self) -> None:
        pass

    def run(self):
        raise NotImplementedError('Experiment has no implementation for run')
    
class BenchmarkExperiment(Experiment):

    def __init__(self, benchmark: Benchmark, optimizer: Optimizer, seed=0) -> None:
        super().__init__()
        self.benchmark = benchmark
        self.optimizer = optimizer
        self.seed = seed

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

    def __init__(self, benchmark: Benchmark, optimizer: Optimizer, prior_task_log) -> None:
        super().__init__()
        self.benchmark = benchmark
        self.optimizer = optimizer
        self.histories = {}
        self._prior_task_log = prior_task_log

    def save(self, file):
        df_dict = {}
        # get all variables encountered during optimization
        vars = []
        for t, run_history in self.histories.items():
            for cfg, res, iteration in run_history:
                vars += list(cfg.keys())
        vars = list(set(vars))

        for t, run_history in self.histories.items():
            res_cols = BenchQueryResult.SUPPORTED_METRICS
            for cfg, res, iteration in run_history:
                for k in vars:
                    if k in cfg:
                        v = cfg[k]
                    else: 
                        v = None
                    if k not in df_dict.keys():
                        df_dict[k] = [v]
                    else:
                        df_dict[k].append(v)
                
                for k in res_cols:
                    if k not in df_dict.keys():
                        df_dict[k] = [res[k]]
                    else:
                        df_dict[k].append(res[k])
                
                if 'iter' not in df_dict and 'task' not in df_dict:
                    df_dict['iter'] = [iteration]
                    df_dict['task'] = [t]
                else:
                    df_dict['iter'].append(iteration)
                    df_dict['task'].append(t)

        df = pd.DataFrame.from_dict(df_dict)
        df.to_csv(file)

    def load_prior_hpo_logs(self):
        raise NotImplementedError('Must be implemented by child class')

    def load_imputed_prior_hpo_runs(self):
        """
            Most of our baselines (all except for MHPD) can only handle homogeneous search spaces.
            To keep them as a baseline and to demonstrate that simple data imputation hacks are not enough
            to use them in heterogeneous search spaces, we build a dataset containing all variables in found
            the logs. Missing values are imputed by the median of the corresponding variable.
        """
        logs = self.load_prior_hpo_logs()
        df = pd.concat(logs)
        performances = df['val_performance']
        hp_df = df[NO_HYPERPARAMETERS]
        for c in hp_df.columns:
            if df[c].isna().any():
                median = df[c].median()
                hp_df.fillna(median, inplace=True)
        
        hp_df = pd.concat((hp_df, performances), axis=1)
        return hp_df