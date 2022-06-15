import torch
from torch import nn
from model.encoder import Encoder
from model.decoder import Decoder
from typing import Tuple

class VAE(nn.Module):
    """ Wrapper class for the Variational Autoencoder """
    
    def __init__(self, d_in: int) -> None:
        """ Params:
                - d_in: dimension of input vector
        """
        super(VAE, self).__init__()
        
        self.encoder = Encoder(d_in)
        self.decoder = Decoder(d_in)
        
    def reparameterize(self, mu: torch.FloatTensor, log_var: torch.FloatTensor) -> torch.FloatTensor:
        """ Making gradient descent applicable to the stochastic sampling process.
            More info here: https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick
            Params:
                - mu: approximated mean by last encoder layers
                - log_var: approximated logarithmic variance by last encoder layers
        """
        if self.training:
            std = log_var.mul(0.5).exp_() # calculate the standart deviation
            # sample from normal distribution and save in independent variable epsilon
            epsilon = torch.autograd.Variable(std.data.new(std.size()).normal_())
            return epsilon.mul(std).add_(mu) # distribution -> tensor
        else:
            return mu
    
    def forward(self, feature_tensor: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """ Params:
                - feature_tensor: input feature vecor from the dataset
        """
        mu, log_var = self.encoder(feature_tensor)
        x = self.reparameterize(mu, log_var)
        return self.decoder(x), mu, log_var
        
