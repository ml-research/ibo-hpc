import torch
from torch import nn

class Encoder(nn.Module):
    """ NN for learning projection """
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=50):
        super().__init__()
        self._encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self._encoder(x)


class BLR:
    """ Bayesian linear regression """
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta  = beta
    
    def fit(self, phi, y):
        S_inv_prior = self.alpha * torch.eye(phi.shape[1]).to(phi.device)
        S_inv       = S_inv_prior + self.beta * phi.t() @ phi
        S           = torch.inverse(S_inv)
        m           = self.beta * S @ phi.t() @ y
        
        self.S = S
        self.m = m
        return self
    
    def predict_with_nll(self, phi, y=None):
        mu  = phi @ self.m
        sig = 1 / self.beta + ((phi @ self.S) * phi).sum(dim=-1)
        if y is not None:
            nll = ((y - mu).pow(2).sum() / sig).mean() + sig.log().mean()
            return mu, sig, nll
        else:
            return mu, sig