import torch
from torch import nn

class Decoder(nn.Module):
    """ Decoder of the VAE """
    
    def __init__(self, d_in: int, d_h1: int = 50, d_h2: int = 12, d_latent: int = 32) -> None:
        """ Params:
                - d_in: dimension of input vector
                - d_h1: dimension of first hidden layer
                - d_h: dimension of intermediary hidden layers
                - d_latent: dimension of the embedding in latent space
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            # sampling vector
            nn.Linear(d_latent, d_latent),
            nn.BatchNorm1d(num_features=d_latent),
            nn.LeakyReLU(),
            nn.Linear(d_latent, d_h2),
            nn.BatchNorm1d(num_features=d_h2),
            nn.LeakyReLU(),
            # decoding vector
            nn.Linear(d_h2, d_h2),
            nn.BatchNorm1d(num_features=d_h2),
            nn.LeakyReLU(),
            nn.Linear(d_h2, d_h1),
            nn.BatchNorm1d(num_features=d_h1),
            nn.LeakyReLU(),
            nn.Linear(d_h1, d_in),
            nn.BatchNorm1d(num_features=d_in),
        ])
        
    def forward(self, distribution: torch.FloatTensor) -> torch.FloatTensor:
        """ Params:
                - distribution_tensor: distribution tensor from the embedding space
        """
        z = distribution
        for l in self.layers:
            z = l(z)
        return z
    