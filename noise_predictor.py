import torch.nn as nn

class NoisePredictor(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(NoisePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        return self.net(x)
    
    

# predicts x_t-1 from x_t 
def reverse_sampling(x_t, alphas, alphas_cumprod, betas, model):
    
    
    pred_noise = model(x_t)