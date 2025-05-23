from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
import torch

class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(ard_num_dims=train_x.shape[1]))
        if lengthscale is not None:
            self.covar_module.base_kernel.lengthscale = lengthscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    

def train_gp(model, likelihood, train_x, train_y, device, training_iter=20000, mll=None, log_loss=False):
    # Find optimal model hyperparameters
    if not torch.device('cpu') == device:
        model = model.to(device)
        likelihood = likelihood.to(device)

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    if mll is None:
        mll = ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x.to(device))
        # Calc loss and backprop gradients
        loss = -torch.mean(mll(output, train_y.to(device)))
        loss.backward()
        if log_loss:
            print('Iter %d/%d - Loss: %.3f' % (
                i + 1, training_iter, loss.item()
            ))
        optimizer.step()

    return model

class SVGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
