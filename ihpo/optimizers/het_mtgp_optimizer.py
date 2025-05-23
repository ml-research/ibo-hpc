from .optimizer import Optimizer
from ..search_spaces import SearchSpace
import numpy as np
import pandas as pd
import torch
import botorch
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from ..utils import ConfigurationScaleOneHotTrnasform
from ..utils.gaussian_processes import HeterogeneousMTGP
from ..consts import NO_HYPERPARAMETERS

class HetMTGPOptimizer(Optimizer):
    
    def __init__(self, search_space: SearchSpace, objective, transfer_learning_evaluation_files, 
                 y_transform, seed=0, gpu=0, iters=100, num_start_samples=5) -> None:
        self.transfer_learning_evaluation_files = transfer_learning_evaluation_files
        self.y_transform = y_transform
        self.config_space_transform = ConfigurationScaleOneHotTrnasform(search_space)
        self._iters = iters
        self._gpu = torch.device(f'cuda:{gpu}') if gpu != -1 else torch.device('cpu')
        self._task_Xs = []
        self._task_ys = []
        self._hp_indices = []
        self._history = []
        self._num_start_samples = num_start_samples
        self.prepare_data()
        super().__init__(search_space, objective, seed)

    def prepare_data(self):
        """
            Here we load the task data and perform some pre-processing required for MTGPs.
            First, we collect the set of all hyperparameters that were seen in former tasks.
            Then, we construct an index set indicating the features/hyperparameters that exist
            in the search space of each task. In parallel, we collect all the features and 
            evaluation scores from former tasks.
            TODO: apply transform!
        """
        no_hyperparameters_wo_val_score = NO_HYPERPARAMETERS + ['val_performance']
        dfs, hps = [], []
        for file in self.transfer_learning_evaluation_files:
            df = pd.read_csv(file, index_col=0)
            df = df[NO_HYPERPARAMETERS]
            # Separate numerical and categorical columns
            numerical_cols = df.select_dtypes(include=['number']).columns
            categorical_cols = df.select_dtypes(exclude=['number']).columns

            # Scale numerical columns
            scaler = MinMaxScaler()
            scaled_numerical = pd.DataFrame(
                scaler.fit_transform(df[numerical_cols]),
                columns=numerical_cols,
                index=df.index
            )

            # One-hot encode categorical columns
            encoded_categorical = pd.get_dummies(df[categorical_cols], prefix=categorical_cols)

            # Combine processed columns in the original order
            processed_columns = []
            for col in df.columns:
                if col != 'val_performance':
                    if col in numerical_cols:
                        processed_columns.append(scaled_numerical[col])
                    elif col in categorical_cols:
                        processed_columns.append(encoded_categorical.filter(like=col))

            df = pd.concat(processed_columns, axis=1)
            dfs.append(df)
            task_hps = [c for c in df.columns if c not in no_hyperparameters_wo_val_score]
            hps += task_hps

        # add hyperparameters of current search space
        curr_task_hps = []
        for name, val in self.search_space.get_search_space_definition():
            if val['dtype'] == 'str' or val['dtype'] == 'cat':
                # if it's a categorical HP, follow pandas naming convention and use [NAME]_[VAL]
                for al in val['allowed']:
                    curr_task_hps.append(name + '_' + str(al))
            else:
                curr_task_hps.append(name)
        hps += curr_task_hps

        for df in dfs:
            feats = [c for c in df.columns if c not in no_hyperparameters_wo_val_score]
            X, y = df[feats].to_numpy(), df['val_performance'].to_numpy()
            hp_indices = [hps.index(c) for c in feats]
            self._task_Xs.append(torch.from_numpy(X))
            self._task_ys.append(torch.from_numpy(y).reshape(-1, 1))
            self._hp_indices.append(hp_indices)
        
        # lastly, add hyperparameter indices of current search space parameters
        curr_feats = list(self.search_space.get_search_space_definition().keys())
        curr_feats_idx = [hps.index(c) for c in curr_feats]
        self._hp_indices.append(curr_feats_idx)
        
        
    def optimize(self):
        """
            Optimization loop maximizing some objective function (BO).
            Here, the pretrained MLP is used to warm-start a GP run.
            Then, in each iteration, the MLP predictions are used to provide probabilities
            of the GP lengthscale parameter for each dimension of the search space. During
            optimization of the GP, these probabilities are used to weight the likelihood of
            the data (see Eq. 5 in https://dl.acm.org/doi/10.1145/3534678.3539255).
        """

        # 2. warmstart BO 
        # first, draw 5 random samples from search space and evaluate them to initialize GP
        configurations = self.search_space.sample(size=self._num_start_samples)
        evaluations = [self.objective(hp) for hp in configurations]
        for i, (cfg, ev) in enumerate(zip(configurations, evaluations)):
            self._history.append((cfg, ev, i))
        val_perfs = [e.val_performance for e in evaluations]
        transformed_cfgs = [self.config_space_transform.transform_configuration(cfg) for cfg in configurations]
        X = torch.from_numpy(np.array(transformed_cfgs))
        y = self.y_transform(np.array(val_perfs))
        y = torch.from_numpy(y).reshape(-1, 1)

        # add first n samples to the history
        for i, (cfg, ev) in enumerate(zip(configurations, evaluations)):
            self._history.append((cfg, ev, i))

        full_feature_dim = len(self._hp_indices)
        # perform BO
        for i in range(self._iters):
            if i % 10 == 0 and i > 0:
                evals = [e.val_performance for _, e, _ in self._history]
                print(f"Iter {i}/{self._iters} \t Incumbent: {max(evals)}")

            model = HeterogeneousMTGP(
                train_Xs=[X, *self._task_Xs],
                train_Ys=[y, *self._task_ys],
                train_Yvars=None,
                feature_indices=self._hp_indices,
                full_feature_dim=len(self._hp_indices),
                input_transform=Normalize(
                    full_feature_dim + 1, indices=list(range(full_feature_dim))
                ),
                outcome_transform=Standardize(m=1),
                use_scale_kernel=False,
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            botorch.fit.fit_gpytorch_mll(mll)

            logNEI = ExpectedImprovement(model=model, best_f=y.max())

            # since all dimensions are bound between 0 and 1, we can proivde these as bounds
            bounds = torch.stack([torch.zeros(X.shape[1]), torch.ones(X.shape[1])]).to(torch.double)
            candidate, acq_value = optimize_acqf(
                logNEI, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
            )

            # apply inverse transform of configuration
            cfg = self.config_space_transform.inv_transform_configuration(candidate.flatten())

            eval_perf = self.objective(cfg)
            X = torch.cat((X, candidate), dim=0)
            y = torch.cat((y.flatten(), torch.tensor([eval_perf.val_performance]).to(torch.float32))).reshape(-1, 1)

            self._history.append((cfg, eval_perf, i + self._num_start_samples))

        performances = np.array([perf.val_performance for _, perf, _ in self._history])
        max_idx = int(np.argmax(performances))
        return self._history[max_idx][:2]
        

    @property
    def history(self):
        return self._history
    

