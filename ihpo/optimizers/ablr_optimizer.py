import torch.utils
from .optimizer import Optimizer
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import scipy
from ..utils import load_task_files, ConfigurationScaleOneHotTrnasform
from ..consts import NO_HYPERPARAMETERS
from ..utils.ablr import BLR, Encoder

torch.set_default_tensor_type('torch.DoubleTensor')

class ABLROptimizer(Optimizer):

    def __init__(self, search_space, objective, transfer_learning_evaluation_files, 
                 seed=0, gpu=-1, iters=100, num_start_samples=5, num_ei_samples=100,
                 pretrain_epochs=30, fine_tune_epochs=5) -> None:
        super().__init__(search_space, objective, seed)
        self.transfer_learning_evaluation_files = transfer_learning_evaluation_files
        self._iters = iters
        self._gpu = torch.device(f'cuda:{gpu}') if gpu > -1 else torch.device('cpu')
        self._history = []
        self._num_start_samples = num_start_samples
        self._pretrain_epochs = pretrain_epochs
        self._fine_tune_epochs = fine_tune_epochs
        self._num_ei_samples = num_ei_samples
        num_dimensions = len(self.search_space.get_search_space_definition())
        self.encoder = Encoder(input_dim=num_dimensions, hidden_dim=100).to(self._gpu)
        self.transform = ConfigurationScaleOneHotTrnasform(search_space)
        self.prepare_data()
        self.alphas  = nn.Parameter(torch.zeros(len(self.transfer_learning_evaluation_files) + 1)).to(self._gpu) # One alpha per historic task
        self.betas   = nn.Parameter(torch.zeros(len(self.transfer_learning_evaluation_files) + 1)).to(self._gpu) # One beta per historic task
        #self.task_alphas  = nn.Parameter(torch.zeros(1)).to(self._gpu) # alpha for new task
        #self.task_betas   = nn.Parameter(torch.zeros(1)).to(self._gpu) # bedta for new task

        params = list(self.encoder.parameters()) + [self.alphas, self.betas]
        self.opt    = torch.optim.LBFGS(params, lr=0.05, max_iter=self._pretrain_epochs)
        self.opt.step(self.fit_surrogate)

    def prepare_data(self):

        self.data_sets = []

        for task, files in self.transfer_learning_evaluation_files.items():
            dfs = load_task_files(files)
            X, y = self._preprocess_data(dfs)
            self.data_sets.append((X, y))

    def fit_surrogate(self):
        """
            Pre-train ABLR. The idea is to train a bayesian linear regression with a non-linear
            kernel parameterized by a neural network. The NN acts like an encoder and tries to learn
            a representation that captures features of previous optimization trajectories.
        """

        total_nll = 0.0
        total_mse = 0.0
        self.opt.zero_grad()
        for idx, (X, y) in enumerate(self.data_sets):
            alpha, beta = 10 ** self.alphas[idx], 1 / 10 ** self.betas[idx]
            X, y = X.to(self._gpu), y.to(self._gpu)

            phi = self.encoder(X)

            blr = BLR(alpha=alpha, beta=beta)
            blr = blr.fit(phi, y)
            mu, sig, nll = blr.predict_with_nll(phi, y)
            total_nll += nll
            total_mse += ((mu - y) ** 2).mean()

            if idx == len(self.transfer_learning_evaluation_files):
                # then we train on the actual task
                self.blr = blr
        
        total_nll /= len(self.transfer_learning_evaluation_files)
        total_mse /= len(self.transfer_learning_evaluation_files)
        total_nll.backward()
        
        torch.nn.utils.clip_grad.clip_grad_norm_([self.alphas, self.betas], 1.)

        print(f"MSE={total_mse}")

        return float(total_nll)

    def optimize(self):

        # sample some points randomly as "warm start" (ensure fair comparison)
        configs = self.search_space.sample(size=self._num_start_samples)
        config_vec = np.array([self.transform.transform_configuration(cfg) for cfg in configs])

        evals = [self.objective(cfg)for cfg in configs]
        val_scores = np.array([e.val_performance for e in evals])
        num_warmstart_cfgs = len(evals)
        self._history += [(cfg, e, i) for i, (cfg, e) in enumerate(zip(configs, evals))]

        for i in range(self._iters):

            self.x_ = torch.from_numpy(config_vec).to(self._gpu)
            self.y_ = torch.from_numpy(val_scores).to(self._gpu)

            if len(self.data_sets) == len(self.transfer_learning_evaluation_files):
                self.data_sets.append((self.x_, self.y_))
            else:
                self.data_sets[-1] = (self.x_, self.y_)

            self.alphas  = nn.Parameter(torch.zeros(len(self.transfer_learning_evaluation_files) + 1)).to(self._gpu) # One alpha per historic task
            self.betas   = nn.Parameter(torch.zeros(len(self.transfer_learning_evaluation_files) + 1)).to(self._gpu) # One beta per historic task
            params = list(self.encoder.parameters()) + [self.alphas, self.betas]
            self.opt    = torch.optim.LBFGS(params, lr=0.05, max_iter=self._pretrain_epochs)
            self.opt.step(self.fit_surrogate)

            #mu, sig, = self.blr.predict_with_nll(self.phi)
            #print(mu)
            #print(self.y_)
            #print(f"MSE={torch.mean((mu - self.y_)**2)}")

            suggestion = self._ei(val_scores.max())

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
        # check all dfs have same set of columns
        if len(dfs) > 1:
            cols = [set(list(df.columns)) for df in dfs]
            check_cols = cols[0]
            intersection = check_cols.intersection(*cols[1:])
            assert intersection == check_cols
        
        df = pd.concat(dfs)
        hyperparams = [col for col in df.columns if col not in NO_HYPERPARAMETERS]
        #if self._hyperparam_select != 'all' and isinstance(self._hyperparam_select, list):
        #    hyperparams = [h for h in hyperparams if h in self._hyperparam_select]
        fdf = df[hyperparams]
        y = df['val_performance'].to_numpy()
        #scaler = MinMaxScaler()
        #y = scaler.fit_transform(y.reshape(-1, 1)).flatten()
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

        return torch.from_numpy(X), torch.from_numpy(y)

    @property
    def history(self):
        return self._history
    

    def _ei(self, y_star):
        eic_samples = self.search_space.sample(size=self._num_ei_samples)
        eic_samples = np.array([self.transform.transform_configuration(cfg) for cfg in eic_samples])
        phi = self.encoder(torch.from_numpy(eic_samples).to(self._gpu))
        mu, sig = self.blr.predict_with_nll(phi)
        mu, sig = mu.flatten().detach().cpu().numpy(), sig.flatten().detach().cpu().numpy()
        improve =  np.abs(y_star) - 0.01 - np.abs(mu)
        scaled = improve / sig
        cdf = scipy.stats.norm.cdf(scaled)
        pdf = scipy.stats.norm.pdf(scaled)
        exploit = improve * cdf
        explore = sig * pdf
        best = np.argmax(explore + exploit).item()
        # return best
        return eic_samples[best]