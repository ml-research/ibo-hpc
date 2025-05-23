from typing import Dict
from .benchmark import BaseBenchmark, BenchQueryResult
from ..search_spaces import FCNetSearchSpace
from ..consts import FCNET_TASK_FILE_NAME_MAPPING, FCNET_MIN_MAX_VALUES
import numpy as np
import h5py
import os
import json

class FCNetBenchmark(BaseBenchmark):

    def __init__(self, task, save_dir='./benchmark_data/fcnet_tabular_benchmarks/') -> None:
        super().__init__()
        self._search_space = FCNetSearchSpace()
        self._file_path = save_dir
        self.task = task
        self._FILE = FCNET_TASK_FILE_NAME_MAPPING[task]
        path = os.path.join(self._file_path, self._FILE)
        self.data = h5py.File(path)

    def query(self, cfg: Dict, budget=None) -> BenchQueryResult:
        rand_seed = np.random.randint(0, 4)
        transformed_cfg = {}
        for k, v in cfg.items():
            if isinstance(v, np.integer):
                transformed_cfg[k] = int(v)
            elif isinstance(v, np.floating):
                transformed_cfg[k] = float(v)
            elif isinstance(v, np.str_):
                transformed_cfg[k] = str(v)
            else:
                transformed_cfg[k] = v

        cfg_str = json.dumps(transformed_cfg)

        try:
            epoch = budget if budget is not None else -1
            val_res = self.data[cfg_str]['valid_mse'][:, epoch][rand_seed]
            test_res = self.data[cfg_str]['final_test_error'][rand_seed]
        except KeyError:
            # if KeyError, then the configuration is not in the tabular benchmark. Return highest MSE TODO.
            val_res = FCNET_MIN_MAX_VALUES[self.task][1]
            test_res = FCNET_MIN_MAX_VALUES[self.task][1]
                
        # FCNet is defined on regression tasks minimizing MSE. Since our optimizers maximize objectives, we have to negate here to minimize MSE.
        return BenchQueryResult(-1., -val_res, -test_res)
        
    
    def get_min_and_max(self):
        configs, ve = [], []
        for k in self.data.keys():
            configs.append(json.loads(k))
            ve.append(np.mean(self.data[k]["valid_mse"][:, -1]))

        return np.min(ve), np.max(ve)
    
    @property
    def search_space(self):
        return self._search_space