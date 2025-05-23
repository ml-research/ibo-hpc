import torch
import torch.nn as nn

class GammaParamaeterActivation(nn.Module):
    def __init__(self, min_k, max_k, min_theta, max_theta):
        super(GammaParamaeterActivation, self).__init__()
        self.min_k = min_k
        self.max_k = max_k
        self.min_theta = min_theta
        self.max_theta = max_theta
    
    def forward(self, x):
        """
        Expects an nx2 input vector where first dimension corresponds to k and second dimension corresponds to theta of a Gamma distribution.

        Args:
            x (torch.FloatTensor): Output of NN

        Returns:
            torch.FloatTensor: Squeezed values between given intervals for k and theta.
        """
        x_k = (self.max_k - self.min_k) * torch.sigmoid(x[:, 0]) + self.min_k
        x_theta = (self.max_theta - self.min_theta) * torch.sigmoid(x[:, 1]) + self.min_theta
        return torch.stack((x_k, x_theta), dim=1)