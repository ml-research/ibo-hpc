from .optimizer import Optimizer
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from ..utils import load_task_files, ConfigurationScaleOneHotTrnasform
from ..consts import NO_HYPERPARAMETERS
from ..utils.ftdkl import FTDKLModel

class FTDKLOptimizer(Optimizer):

    def __init__(self, search_space, objective, transfer_learning_evaluation_files, 
                 seed=0, gpu=-1, iters=100, checkpoint='./ftdkl_checkpoints/', model_name='model', num_start_samples=5,
                 pretrain_epochs=10000, fine_tune_epochs=1000) -> None:
        super().__init__(search_space, objective, seed)
        self.transfer_learning_evaluation_files = transfer_learning_evaluation_files
        self._iters = iters
        self._gpu = torch.device(f'cuda:{gpu}') if gpu > -1 else torch.device('cpu')
        self._history = []
        self._checkpoint_dir = checkpoint
        self._model_name = model_name
        self._num_start_samples = num_start_samples
        self._pretrain_epochs = pretrain_epochs
        self._fine_tune_epochs = fine_tune_epochs
        self.transform = ConfigurationScaleOneHotTrnasform(search_space)
        self.pretrain()

    def pretrain(self):
        """
            Pre-train FTDKL. The idea is to train a deep kernel similar to what is done in MAML.
            Then, with only a few gradient steps, the deep kernel GP will fit well to the new obtained
            task.
        """
        search_space_def = self.search_space.get_search_space_definition()
        n_cont_features = len([v for v in search_space_def.values() if v['dtype'] == 'float'])
        cat_cardinalities = [len(v['allowed'] for v in search_space_def.values() if v['dtype'] != 'float')]

        dfs = []
        for task, files in self.transfer_learning_evaluation_files.items():
            task_dfs = load_task_files(files)
            dfs += task_dfs
            
        X, y, hyperparameters = self._preprocess_data(dfs)
            
        data = np.concatenate((X, y), axis=1)
        self.model = FTDKLModel(data, hyperparameters, n_cont_features, cat_cardinalities, 512, self._gpu, self._checkpoint_dir, self._model_name)
        self.model.train()


    def optimize(self):

        # sample some points randomly as "warm start" (follows FSBO/HPO-B appraoch)
        configs = self.search_space.sample(size=self._num_start_samples)
        config_vec = np.array([self.transform.transform_configuration(cfg) for cfg in configs])

        evals = [self.objective(cfg)for cfg in configs]
        val_scores = np.array([e.val_performance for e in evals])
        num_warmstart_cfgs = len(evals)
        self._history += [(cfg, e, i) for i, (cfg, e) in enumerate(zip(configs, evals))]

        for i in range(self._iters):
            # get new observation from Deep Kernel GP
            ei_configs = self.search_space.sample(size=1000)
            ei_configs = np.array([self.transform.transform_configuration(cfg) for cfg in ei_configs])
            suggestion_idx = self.model.observe_and_suggest(config_vec, val_scores, ei_configs)
            suggestion = ei_configs[suggestion_idx]

            # evaluate suggestion
            cfg = self.transform.inv_transform_configuration(torch.from_numpy(suggestion).flatten())
            evaluation = self.objective(cfg)

            self._history.append((cfg, evaluation, i + num_warmstart_cfgs))
            score = evaluation.val_performance

            # update dataset
            val_scores = np.concatenate((val_scores, np.array([score])))
            config_vec = np.concatenate((config_vec, suggestion.reshape(1, -1)))
        
        max_idx = np.argmax(val_scores).flatten()[0]
        cfg = self.transform.inv_transform_configuration(torch.from_numpy(config_vec[max_idx]))
        return cfg, val_scores[max_idx]

    def _preprocess_data(self, dfs):
        """
            Transform categorical features into a one-hot-encoding and scale each x-value.
            NOTE: Scaling y is not required here as FSBO does this for every batch automatically.
                This is to be scale-invariant.
        """

        for df in dfs:
            scaler = StandardScaler()
            df['val_performance'] = scaler.fit_transform(df['val_performance'].to_numpy().reshape(-1, 1)).flatten()

        df = pd.concat(dfs, axis=0, ignore_index=True).fillna(0)
        hyperparams = [col for col in df.columns if col not in NO_HYPERPARAMETERS]
        #if self._hyperparam_select != 'all' and isinstance(self._hyperparam_select, list):
        #    hyperparams = [h for h in hyperparams if h in self._hyperparam_select]
        fdf = df[hyperparams]
        y = df['val_performance'].to_numpy()
        fdf = fdf.drop(columns=['val_performance'])
        fdf = fdf.reset_index()

        num_samples = len(fdf)

        fdf_dict = fdf.to_dict()
        transformed_hps = []
        for i in range(num_samples):
            cfg_dict = {}
            for k, v in fdf_dict.items():
                cfg_dict[k] = v[i]

            transformed_cfg = self.transform.transform_configuration(cfg_dict)
            transformed_hps.append(transformed_cfg)
        
        X = np.array(transformed_hps)

        return torch.from_numpy(X), torch.from_numpy(y), hyperparams

    @property
    def history(self):
        return self._history