import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import os 
sys.path.append(os.getcwd())

from model.vae import VAE
from model.utils import uniform_weights_init, HYPERPARAMS, elbo


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


final_features = np.load("data/final_features.npy")
dimension_input_data = final_features.shape[1] # = 206 (243 in original paper)

model = VAE(d_in=dimension_input_data).to(device)
model.apply(uniform_weights_init)
optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMS["learning_rate"])
trainloader=DataLoader(dataset=final_features, batch_size=HYPERPARAMS["batch_size"], pin_memory=True)
loss_func = elbo()
train_losses = []
val_losses = []
mu_output = []
log_var_output = []


def train(epoch):
    model.train()
    train_loss = 0
    for data in trainloader:
        data = data.to(device)
        optimizer.zero_grad()
        decoded_batch, mu, log_var = model(data)
        loss = loss_func(decoded_batch, data, mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if epoch % 200 == 0:        
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(trainloader.dataset)))
        train_losses.append(train_loss / len(trainloader.dataset))

for epoch in range(1, HYPERPARAMS["epochs"] + 1):
    train(epoch)


with torch.no_grad():
    for i, (data) in enumerate(trainloader):
        data = data.to(device)
        optimizer.zero_grad()
        decoded_batch, mu, log_var = model(data)

        mu_tensor = mu   
        mu_output.append(mu_tensor)
        mu_result = torch.cat(mu_output, dim=0)

        log_var_tensor = log_var   
        log_var_output.append(log_var_tensor)
        log_var_result = torch.cat(log_var_output, dim=0)

pred = mu_result.cpu().detach().numpy() 
pred.shape
np.save('data/pred_1000_epochs.npy', pred)
