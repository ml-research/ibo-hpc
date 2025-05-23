from .optimizer import Optimizer
from ..search_spaces import SearchSpace
import numpy as np
import torch
from gpytorch.likelihoods import GaussianLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from ..utils import train_gp, ConfigurationScaleOneHotTrnasform

class GPOptimizer(Optimizer):
    
    def __init__(self, search_space: SearchSpace, objective, y_transform, seed=0, gpu=0, iters=100, num_start_samples=5) -> None:
        self.y_transform = y_transform
        self.config_space_transform = ConfigurationScaleOneHotTrnasform(search_space)
        self._iters = iters
        self._gpu = torch.device(f'cuda:{gpu}') if gpu != -1 else torch.device('cpu')
        self._context_x = []
        self._context_y = []
        self._history = []
        self._num_start_samples = num_start_samples
        super().__init__(search_space, objective, seed)

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
        likelihood = GaussianLikelihood()

        # add first n samples to the history
        for i, (cfg, ev) in enumerate(zip(configurations, evaluations)):
            self._history.append((cfg, ev, i))

        # perform BO
        for i in range(self._iters):
            if i % 10 == 0 and i > 0:
                evals = [e.val_performance for _, e, _ in self._history]
                print(f"Iter {i}/{self._iters} \t Incumbent: {max(evals)}")
            gp = SingleTaskGP(train_X=X, train_Y=y, input_transform=Normalize(d=X.shape[1]), outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(likelihood, gp)
            gp = train_gp(gp, likelihood, X, y, device=self._gpu, training_iter=1000, mll=mll, log_loss=False)

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
    

