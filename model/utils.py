import numpy as np
import torch
from torch import nn
from typing import Dict, Any

HYPERPARAMS: Dict[str, Any] = {
    "learning_rate" : 1e-3,
    "epochs" : 1000,
    "log_interval" : 50,
    "batch_size" : 1024,
}

class elbo(nn.Module):
    """ Evidence lower bound loss func
        More info: https://lilianweng.github.io/posts/2018-08-12-vae/#loss-function-elbo
    """
    
    def __init__(self) -> None:
        super(elbo, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
    
    def forward(self, decoded_x: torch.FloatTensor, original_x: torch.FloatTensor, mu: torch.FloatTensor, log_var: torch.FloatTensor):
        loss_MSE = self.mse(decoded_x, original_x)
        loss_KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return loss_MSE + loss_KLD

def uniform_weights_init(m):
    """ Uniform weight initializer """
    
    if type(m) == nn.Linear: # for every linear layer
        n = m.in_features # get the number of the inputs
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)