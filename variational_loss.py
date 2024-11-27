import torch.nn as nn
import torch.nn.functional as F
import torch


class VariationalLoss(nn.Module):
    
    def __init__(self, beta=1):
        super(VariationalLoss, self).__init__()
        self.beta = beta
        
    def forward(self, x_hat, x, mu, logsigma2):
        mse_loss = F.mse_loss(x_hat, x, reduction='mean') 
        
        # prior = -1/2 * sum(1+logsigma2-mu2-sigma2)
        sigma2 = torch.exp(logsigma2)
        mu2 = mu**2
        prior_loss = torch.sum(1 + logsigma2 - mu2 - sigma2, dim=1)
        prior_loss = -0.5 * (prior_loss)
        prior_loss = self.beta * prior_loss.mean()
        
        return mse_loss + prior_loss
    

if __name__ == '__main__':
    loss_fn = VariationalLoss(beta=1.0)
    x_hat = torch.randn(16, 1, 512, 512)
    x = torch.randn(16, 1, 512, 512)
    mu = torch.randn(16, 256)
    logsigma2 = torch.randn(16, 256)
    print(f"Variational loss value {loss_fn(x_hat, x, mu, logsigma2)}")