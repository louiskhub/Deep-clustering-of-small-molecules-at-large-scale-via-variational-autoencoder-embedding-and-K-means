import torch
from torch import nn
from typing import Tuple

class Encoder(nn.Module):
    """ Encoder of the VAE """
    
    def __init__(self, d_in: int, d_h1: int = 50, d_h2: int = 12, d_latent: int = 32) -> None:
        """ Params:
                - d_in: dimension of input vector
                - d_h1: dimension of first hidden layer
                - d_h: dimension of intermediary hidden layers
                - d_latent: dimension of the embedding in latent space
        """
        super(Encoder, self).__init__()
        # Classic Encoder
        self.layers = nn.ModuleList([
            nn.Linear(d_in, d_h1),
            nn.BatchNorm1d(num_features=d_h1),
            nn.LeakyReLU(),
            nn.Linear(d_h1, d_h2),
            nn.BatchNorm1d(num_features=d_h2),
            nn.LeakyReLU(),
            nn.Linear(d_h2, d_h2),
            nn.BatchNorm1d(num_features=d_h2),
            nn.LeakyReLU(),
            nn.Linear(d_h2, d_latent),
            nn.BatchNorm1d(num_features=d_latent),
            nn.LeakyReLU(),
        ])
        # We need this for the stochastic part
        self.mu = nn.Linear(d_latent, d_latent)         # weights will learn to compute mean
        self.log_var = nn.Linear(d_latent, d_latent)    # weights will learn to compute logarithmic variance
    
    def forward(self, feature_tensor: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """ Params:
                - feature_tensor: input feature vecor from the dataset
        """
        x = feature_tensor
        for l in self.layers:
            x = l(x)
        return self.mu(x), self.log_var(x)
        
