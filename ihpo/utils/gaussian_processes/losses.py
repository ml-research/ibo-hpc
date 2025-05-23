from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
import torch

class MHPDLoss(gpytorch.Module):

    """
        Loss that incorporates log-likelihoods of parameters of GP predicted by a MLP.
        See https://dl.acm.org/doi/10.1145/3534678.3539255 Sec. 4, Eq. 5 for details.
    """

    def __init__(self, gamma: torch.distributions.Gamma, gp, likelihood) -> None:
        super().__init__()
        self.gamma = gamma
        self.emll = ExactMarginalLogLikelihood(likelihood, gp)
        self.model = gp
        self.likelihood = likelihood

    def forward(self, approximate_dist_f, target, **kwargs):
        emll_loss = self.emll(approximate_dist_f, target, **kwargs)
        lengthscale = self.model.covar_module.base_kernel.lengthscale
        return torch.sum(emll_loss) + torch.sum(self.gamma.log_prob(lengthscale))

