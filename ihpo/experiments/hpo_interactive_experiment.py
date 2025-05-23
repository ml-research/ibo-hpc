from .experiment import BenchmarkExperiment
from ..benchmarks import BenchmarkFactory
from ..optimizers import OptimizerFactory

class HPOInteractiveExperiment(BenchmarkExperiment):

    def __init__(self, optimizer_name, intervention_iter=50, intervention_duration=1,
                 intervention_type='good', task=167120, seed=0) -> None:
        self.benchmark_config = {
            'model': 'xgb',
            'task_id': int(task)
        }
        self._intervention_type = intervention_type
        self._intervention_iter = intervention_iter
        self._intervention_duration = intervention_duration
        benchmark = BenchmarkFactory.get_benchmark('hpo', self.benchmark_config)
        self._optimizer_name = optimizer_name   
        self.optimizer_config = self.get_optimizer_config(benchmark)
        optimizer = OptimizerFactory.get_optimizer(optimizer_name, self.optimizer_config)        
        super().__init__(benchmark, optimizer)

    def run(self):
        # register intervention
        intervention = self.get_intervention()
        self.optimizer.intervene(intervention, iter=self._intervention_iter, reset_after=self._intervention_duration)
        config, performance = self.optimizer.optimize()
        if self._optimizer_name == 'pc':
            processed_config = {}
            for name, idx in config.items():
                param_def = self.benchmark.search_space.search_space_definition[name]
                processed_config[name] = param_def['allowed'][int(idx)]
            config = processed_config

        print((config, performance))

    def evaluate_config(self, cfg):
        # if PC is optimizer, we have to discretize search space due to tabular benchmark
        # -> convert discretized configs back to original domain
        if self._optimizer_name == 'pc' and self.benchmark is not None:
            processed_config = {}
            for name, idx in cfg.items():
                param_def = self.benchmark.search_space.search_space_definition[name]
                processed_config[name] = param_def['allowed'][int(idx)]
        else:
            processed_config = cfg
        if self.benchmark is not None:
            res = self.benchmark.query(processed_config)
            return res

    def get_optimizer_config(self, benchmark):
        if self._optimizer_name == 'pibo':
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'n_trials': 1000
            }
        elif self._optimizer_name == 'pc': 
            return {
                'objective': self.evaluate_config,
                'search_space': benchmark.search_space,
                'iterations': 500,
                'samples_per_iter': 20,
                'use_eic': False,
                'eic_samplings': 20
            }
        else:
            raise ValueError(f'No such optimizer: {self._optimizer_name}')
        
    def get_intervention(self):
        metadata = self.benchmark.benchmark.get_meta_information()
        if self._intervention_type == 'good':
            # return useful intervention
            intervention = metadata['global_min']
        else:
            # return harmful intervention
            intervention = metadata['global_max']
        return intervention