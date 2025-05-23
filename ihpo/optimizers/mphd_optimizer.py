from .optimizer import Optimizer
from ..search_spaces import SearchSpace
from ..consts import NO_HYPERPARAMETERS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dists
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from gpytorch.likelihoods import GaussianLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from ..utils import ExactGPModel, train_gp, load_task_files, train_mlp, ConfigurationScaleOneHotTrnasform, GammaParamaeterActivation
from ..utils.gaussian_processes import MHPDLoss
import os 

class MPHDOptimizer(Optimizer):
    
    def __init__(self, search_space: SearchSpace, objective, transfer_learning_evaluation_files, 
                 y_transform, seed=0, gpu=0, iters=100, num_start_samples=5, gp_param_cache_file='./data/mphd_cache/cache.csv') -> None:
        self.transfer_learning_evaluation_files = transfer_learning_evaluation_files
        self.y_transform = y_transform
        self.config_space_transform = ConfigurationScaleOneHotTrnasform(search_space)
        self._iters = iters
        self._gpu = torch.device(f'cuda:{gpu}') if gpu != -1 else torch.device('cpu')
        self._context_x = []
        self._context_y = []
        self._history = []
        self._gp_param_cache_file = gp_param_cache_file
        self._num_start_samples = num_start_samples
        self.pre_train_fit_gps()
        self.pre_train_fit_mlp()
        super().__init__(search_space, objective, seed)

    def pre_train_fit_gps(self):
        """
            Fit a GP for each task seen so far.
            First, prepare the data, i.e. throw all columns away that are no hyperparameters
            Then, construct a context vector as in https://dl.acm.org/doi/10.1145/3534678.3539255 that
            records some statistics about the search space (needed since each search space can be different).
            Then, fit GP and get lengthscale for each hyperparamter. 
            The context-lengthscale pairs will be used by the MLP to predict a lengthscale for a given
            search sapce, thus aiming to warm-start the BO of the actual task.
        """
        if os.path.exists(self._gp_param_cache_file):
            cache = pd.read_csv(self._gp_param_cache_file, index_col=0)
        else:
            directory = './data/mphd_cache/'
            if not os.path.exists(directory):
                os.mkdir(directory)
            cache = pd.DataFrame(columns=['cache_id', 'is_disc', 'is_cont', 'num_cont', 'num_disc', 'l'])

        for task, files in self.transfer_learning_evaluation_files.items():
            dfs = load_task_files(files)
            X, y, ctxts = self._preprocess_data(dfs)

            print(f"Learning {len(X)} GPs...")

            for i in range(len(X)):
                x = X[i]
                targets = y[i].reshape(-1, 1)
                ctxt = ctxts[i]
                file_name = files[i]
                cache_id = f'{task}_{file_name}'

                cached_gp_params_with_context = cache[cache['cache_id'] == cache_id]
                if not cached_gp_params_with_context.empty:
                    length_scale = cached_gp_params_with_context['l'].to_numpy()
                else:
                    likelihood = GaussianLikelihood()
                    #gp = ExactGPModel(x, targets, likelihood)
                    gp = SingleTaskGP(train_X=x, train_Y=targets, input_transform=Normalize(d=x.shape[1]), outcome_transform=Standardize(m=1))

                    gp = train_gp(gp, likelihood, x, targets, device=self._gpu, training_iter=1000, log_loss=False)

                    length_scale = gp.covar_module.base_kernel.lengthscale.reshape(-1, 1).detach().cpu()

                    # add the trained GP parameters into the cache along with the corresponding context vectors
                    for j in range(length_scale.shape[0]):
                        cache_vector = [cache_id] + ctxt[j].tolist() + length_scale[j].tolist()
                        cache.loc[len(cache)] = cache_vector

                self._context_x += ctxt.detach().tolist()
                self._context_y += length_scale.flatten().tolist()

        self._context_x = torch.from_numpy(np.array(self._context_x)).to(torch.float32)
        self._context_y = torch.from_numpy(np.array(self._context_y)).to(torch.float32)
        cache.to_csv(self._gp_param_cache_file)

    def pre_train_fit_mlp(self):
        """
            Pre-train the MLP on context-evaluation score pairs obtained during preprocessing.
            The MLP architecture is given in https://dl.acm.org/doi/10.1145/3534678.3539255.

            Afterwards, the MLP is trained to predict the lenghtscale parameter of a GP 
            s.t. the log-likelihood of a Gamma distribution is maximized from which the lengthscale is sampled.
            The Gamma distirbution's parameters are predicted by the MLP.
        """
        input_size = self._context_x.shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            GammaParamaeterActivation(min_k=0.1, max_k=5e4, min_theta=0.1, max_theta=5e3) # sigmoid with custom bounds, stabalizes training
        ).to(self._gpu, dtype=torch.float32)

        self.mlp = train_mlp(self.mlp, self._context_x, self._context_y, self._gpu, 10000)

        # print approx. distance
        gamma_params = self.mlp(self._context_x.to(self._gpu)) + 1e-7
        gamma_params = gamma_params.detach().cpu()
        c, r = gamma_params[:, 0], gamma_params[:, 1]
        dist = dists.Gamma(c, r)
        samples = dist.sample_n(self._context_y.shape[0])

    def _map_inference(self, dist, num_samples=1000):
        """Perform Monte Carlo optimization to do MAP inference.

        Args:
            dist (torch.dist.Distribution): torch distribution object
            num_samples (int, optional): Number of MC samples. Defaults to 1000.
        """
        # TODO: MAP inference is done here via Monte Carlo, but there might be a closed form or gradient based solution?
        probs = []
        lengthscales = []
        lengthscale = []
        for _ in range(num_samples):
            samples = dist.sample()
            lengthscales.append(samples)
            p = dist.log_prob(samples)
            probs.append(p.numpy())
        probs = torch.from_numpy(np.vstack(probs))
        lengthscales = torch.from_numpy(np.vstack(lengthscales))
        for col in range(probs.shape[1]):
            argmax = probs[:, col].argmax().item()
            lengthscale.append(lengthscales[argmax, col])
        return torch.from_numpy(np.array(lengthscale)).reshape(-1, 1)
    
    def _preprocess_data(self, dfs):
        """
            Preprocess the data for GP and MLP training.
            We perform normalization of all continuous variables and one-hot encoding of all discrete variables
        """
        preprocessed_dfs, targets, ctxts = [], [], []
        for df in dfs:
            hyperparams = [col for col in df.columns if col not in NO_HYPERPARAMETERS]
            fdf = df[hyperparams]
            y = df['val_performance']
            fdf = fdf.drop(columns=['val_performance'])

            # Separate numeric and categorical columns
            numeric_cols = fdf.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = fdf.select_dtypes(include=['object', 'category', 'bool']).columns

            # construct context vector
            cont_indicator = np.array([0] * len(categorical_cols) + [1] * len(numeric_cols))
            disc_indicator = 1 - cont_indicator
            frac_cont = len(numeric_cols) / (len(numeric_cols) + len(categorical_cols))
            frac_disc = len(categorical_cols) / (len(numeric_cols) + len(categorical_cols))
            num_cont = [frac_cont] * (len(numeric_cols) + len(categorical_cols))
            num_disc = [frac_disc] * (len(numeric_cols) + len(categorical_cols))
            context_vector = np.array([cont_indicator, disc_indicator, num_cont, num_disc]).T
            
            # Convert categorical columns to one-hot encoding
            if len(categorical_cols) > 0:
                cols = categorical_cols[0] if len(categorical_cols) == 1 else list(categorical_cols)
                le = LabelEncoder()
                fdf[cols] = le.fit_transform(fdf[cols])

            # Normalize numeric data
            scaler = MinMaxScaler()
            fdf[numeric_cols] = scaler.fit_transform(fdf[numeric_cols])

            # normalize y
            y = self.y_transform(y.to_numpy().flatten())

            preprocessed_dfs.append(torch.from_numpy(fdf.to_numpy())) 
            targets.append(torch.from_numpy(y)) 
            ctxts.append(torch.from_numpy(context_vector))
        return preprocessed_dfs, targets, ctxts
    
    def _construct_context_for_target_task(self):
        search_space_def = self.search_space.get_search_space_definition()
        context = []
        num_disc, num_cont = 0, 0 
        for idx, (name, d) in enumerate(search_space_def.items()):
            if d['dtype'] == 'str':
                for _ in range(len(d['allowed']) - 1):
                    context.append([0, 1])
                    num_disc += 1
            elif d['dtype'] == 'bool':
                context.append([0, 1])
                num_disc += 1
            else:
                context.append([1, 0])
                num_cont += 1

        context = np.array(context)
        cont_indicator = np.array([num_cont / (num_disc + num_cont)] * (num_disc + num_cont)).reshape(-1, 1)
        disc_indicator = np.array([num_disc / (num_disc + num_cont)] * (num_disc + num_cont)).reshape(-1, 1)
        context = np.concatenate([cont_indicator, disc_indicator, context], axis=1).astype(np.float32)
        self.target_context = context
        return self.target_context
    
    def _infer_and_gamma(self, target_ctxt):
        #target_ctxt = self.mlp_scaler_x.transform(target_ctxt)
        target_ctxt = torch.from_numpy(target_ctxt)
        gamma_params = self.mlp(target_ctxt.to(self._gpu)) + 1e-5
        gamma_params = gamma_params.detach().cpu()
        c, r = gamma_params[:, 0], gamma_params[:, 1]
        dist = dists.Gamma(c, r)
        return dist

    def optimize(self):
        """
            Optimization loop maximizing some objective function (BO).
            Here, the pretrained MLP is used to warm-start a GP run.
            Then, in each iteration, the MLP predictions are used to provide probabilities
            of the GP lengthscale parameter for each dimension of the search space. During
            optimization of the GP, these probabilities are used to weight the likelihood of
            the data (see Eq. 5 in https://dl.acm.org/doi/10.1145/3534678.3539255).
        """
        # 1. construct context for new dataset and obtain lengthscale parameter
        target_ctxt = self._construct_context_for_target_task()
        gamma = self._infer_and_gamma(target_ctxt)

        # 2. warmstart BO 
        # first, draw 5 random samples from search space and evaluate them to initialize GP
        configurations = self.search_space.sample(size=self._num_start_samples)
        evaluations = [self.objective(hp) for hp in configurations]
        val_perfs = [e.val_performance for e in evaluations]
        transformed_cfgs = [self.config_space_transform.transform_configuration(cfg) for cfg in configurations]
        X = torch.from_numpy(np.array(transformed_cfgs))
        y = self.y_transform(np.array(val_perfs))
        y = torch.from_numpy(y).reshape(-1, 1)
        likelihood = GaussianLikelihood()

        # add first n samples to the history
        for i, (cfg, ev) in enumerate(zip(configurations, evaluations)):
            self._history.append((cfg, ev, i))

        # 3. perform BO
        for i in range(self._iters):
            if i % 10 == 0 and i > 0:
                evals = [e.val_performance for _, e, _ in self._history]
                print(f"Iter {i}/{self._iters} \t Incumbent: {max(evals)}")
            gp = SingleTaskGP(train_X=X, train_Y=y, input_transform=Normalize(d=X.shape[1]), outcome_transform=Standardize(m=1))
            #gp.covar_module.base_kernel.lengthscale = lengthscales
            mll = MHPDLoss(gamma, gp, likelihood)
            gp = train_gp(gp, likelihood, X, y, device=self._gpu, training_iter=500, mll=mll, log_loss=False)

            logNEI = ExpectedImprovement(model=gp, best_f=y.max(), maximize=True)

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
    

