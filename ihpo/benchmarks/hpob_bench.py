from typing import Dict
from .benchmark import BaseBenchmark, BenchQueryResult
from ..search_spaces import HPOBSearchSpace
from spn.structure.StatisticalTypes import MetaType
import json
import xgboost as xgb
import numpy as np
import os

class HPOBBenchmark(BaseBenchmark):

    def __init__(self, dataset_path, surrogate_path, search_space_id, dataset_id) -> None:
        super().__init__()
        self._search_space = HPOBSearchSpace(dataset_path, search_space_id)
        self._search_space_id = search_space_id
        self._dataset_id = dataset_id
        self._surrogate_path = surrogate_path
        self._dataset_path = dataset_path
        self._prepare_surrogate()

    def _prepare_surrogate(self):
        surrogate_json = 'surrogate-' + self._search_space_id +'-' + self._dataset_id + '.json'
        bst_surrogate = xgb.Booster()
        file = os.path.join(self._surrogate_path, surrogate_json)
        bst_surrogate.load_model(file)
        self.surrogate = bst_surrogate
       
        # compute min and max for scaling
        data_set_file = os.path.join(self._dataset_path, 'meta-test-dataset.json') # use HPOB-v3
        with open(data_set_file, 'r') as f:
            data = json.load(f)
        y = np.array(data[self._search_space_id][self._dataset_id]['y'])
        self._min_prediction = y.min()
        self._max_prediction = y.max()
        
        
    def _prepare_configuration(self, cfg: Dict):
        vector_repr = []
        for name, val in self._search_space.get_search_space_definition().items():
            if val['dtype'] == 'str':
                # one hot encoding
                choices = val['allowed']
                one_hot_vec = np.zeros(len(choices))
                one_idx = [i for i in range(len(choices)) if cfg[name] == choices[i]][0]
                one_hot_vec[one_idx] = 1
                vector_repr += one_hot_vec.tolist()

            elif val['dtype'] == 'float' or val['dtype'] == 'int':
                # scale between 0 and 1
                #min_, max_ = val['min'], val['max']
                #scaled_hp = (cfg[name] - min_) / (max_ - min_)
                vector_repr.append(cfg[name])
        return np.array(vector_repr)
    
    def get_min_and_max(self):
        return self._min_prediction, self._max_prediction

    def query(self, cfg: Dict, budet=None) -> BenchQueryResult:
        hp_repr = self._prepare_configuration(cfg)
        dim = len(hp_repr)
        x_q = xgb.DMatrix(hp_repr.reshape(-1, dim))
        y_pred = self.surrogate.predict(x_q)
        return BenchQueryResult(
            y_pred[0],
            y_pred[0],
            y_pred[0] # NOTE: HPO-B does not provide separate values for train, val and test-set
        )

    @property
    def search_space(self):
        return self._search_space