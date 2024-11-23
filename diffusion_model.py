import torch.nn as nn
import torch
import torch.nn.functional as F

class ConditionalLayer(nn.Module):
    def __init__(self, input_dim, feature_dim, n_steps):
        super(ConditionalLayer, self).__init__()
        self.output_dim = feature_dim
        self.linear = nn.Linear(in_features=input_dim, out_features=feature_dim)
        self.embedding = nn.Embedding(num_embeddings=n_steps, embedding_dim=feature_dim)
    
    def forward(self, x, t):
        # x: [B, input_dim]
        linear_out = self.linear(x)
        embed_out = self.embedding(t)
        
        return embed_out.view(-1, self.output_dim) * linear_out # [B, output_dim]
    
class ConditionalModel(nn.Module):
    def __init__(self, x_dim, n_steps, feature_dim):
        super(ConditionalModel, self).__init__()
        self.x_dim = x_dim
        self.input_dim = self.x_dim
        self.output_dim = self.input_dim
        self.feature_dim = feature_dim
        self.n_steps = n_steps + 1 # account for 0 indexed embeddings
        
        self.x_embed_bn = nn.BatchNorm1d(self.feature_dim)
        self.cl1 = ConditionalLayer(input_dim=self.x_dim, feature_dim=self.feature_dim, n_steps=self.n_steps)
        self.bn1 = nn.BatchNorm1d(self.feature_dim)
        self.cl2 = ConditionalLayer(input_dim=self.feature_dim, feature_dim=self.feature_dim, n_steps=self.n_steps)
        self.bn2 = nn.BatchNorm1d(self.feature_dim)
        self.cl3 = ConditionalLayer(input_dim=self.feature_dim, feature_dim=self.feature_dim, n_steps=self.n_steps)
        self.bn3 = nn.BatchNorm1d(self.feature_dim)
        self.l4 = nn.Linear(in_features=self.feature_dim, out_features=self.output_dim)
        
    def forward(self, x_embed, x, t):
        
        x_e = self.x_embed_bn(x_embed)
        out_cl1 = F.softplus(self.bn1(self.cl1(x=x, t=t)))
        modulated_x = x_e * out_cl1
        out_cl2 = F.softplus(self.bn2(self.cl2(x=modulated_x, t=t)))
        out_cl3 = F.softplus(self.bn3(self.cl3(x=out_cl2, t=t)))
        return self.l4(out_cl3)
        
class LabelDenoisingDiffusionModel(nn.Module):
    
    def __init__(self, device_id, x_dim, x_emb_dim, n_steps):
        super(LabelDenoisingDiffusionModel, self).__init__()
        self.x_dim = x_dim
        self.n_steps = n_steps
        self.feature_dim = x_emb_dim
        self.device_id = device_id
        self.betas = self.get_noise_schedule(schedule='linear', num_steps=self.n_steps, start=1e-5, end=1e-2).to(self.device_id)
        self.alphas = 1.0 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alphas = torch.sqrt(1.0 - self.alphas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) # signal contribution
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod) # scaled signal contribution coeff
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod # noise contribution coeff
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(self.one_minus_alphas_cumprod) # scaled noise contribution coeff
        
        self.model = ConditionalModel(x_dim=self.x_dim, n_steps=self.n_steps, feature_dim=self.feature_dim)
        
    def get_noise_schedule(self, schedule='linear', num_steps=1000, start=1e-5, end=1e-2):
    
        if schedule == 'linear':
            betas = torch.linspace(start=start, end=end, steps=num_steps)
            
        return betas
    
    def q_sample(self, x0, t, noise):
        # x0: [B, 10]
        # t: [B, 1]
        # noise: [B, 10]
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(0, t) # [batch_size]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t) # [batch_size]
        
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1)
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def get_diffusion_loss(self, pred, gt):
        return F.mse_loss(pred, gt, reduction='mean')
    
    def forward_t(self, x_embed, x0, t):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0=x0, t=t, noise=noise)
        noise_pred = self.model(x_embed=x_embed, x=xt, t=t)
        return noise_pred, noise
    
    def reverse(self, x_embed, xt):
        # x_embed: [batch, feature_dim]
        # xt: [batch, num_classes]
        
        for t in reversed(range(self.n_steps)):
            t_tens = torch.tensor(t)
            noise_pred = self.model(x_embed=x_embed, x=xt, t=t_tens) # [batch, self.x_dim]
            
            sqrt_one_minus_alphas_t = self.sqrt_one_minus_alphas.gather(0, t_tens) # [1]
            sqrt_alphas_t = self.sqrt_alphas.gather(0, t_tens) # [1]
            
            x_t_minus_1 = torch.divide(xt - sqrt_one_minus_alphas_t * noise_pred, sqrt_alphas_t)
            
            if t > 0:
                beta_t = self.betas.gather(0, t_tens)
                sigma_t = torch.sqrt(beta_t)
                stochastic_sampling_noise = sigma_t * torch.randn_like(xt) # sigma = 1
                x_t_minus_1 = x_t_minus_1 + stochastic_sampling_noise
                
            xt = x_t_minus_1

        x0 = xt
        
        return x0

if __name__ == '__main__':
    
    batch_size = 16
    num_classes = 10
    feature_dim = 256
    n_steps = 1000
    
    diffusion_model = LabelDenoisingDiffusionModel(
        x_dim=num_classes,
        x_emb_dim=feature_dim,
        n_steps=n_steps,
    )
    
    random_label = torch.randint(low=0, high=10, size=(batch_size, num_classes)).float()
    random_feat = torch.randn(batch_size, feature_dim)
    
    t = torch.randint(low=0, high=n_steps, size=(batch_size//2+1, ))
    ts = torch.cat([t, n_steps - t], dim=0)[:batch_size]
    
    # simulate forward process and learning
    pred_noise, gt_noise = diffusion_model.forward_t(x_embed=random_feat, x0=random_label, t=ts)
    print(f"gt_noise {gt_noise.shape} | pred_noise {pred_noise.shape}")
    mse_loss = F.mse_loss(pred_noise, gt_noise, reduction='mean')
    print(f"MSE calculated: {mse_loss}")
    
    # backward process (denoising)
    noisy_label_batch = torch.randint(low=0, high=10, size=(batch_size, num_classes)).float()
    random_feat = torch.randn(batch_size, feature_dim)
    clean_label_batch = diffusion_model.reverse(x_embed=random_feat, xt=noisy_label_batch)
    print(f"Clean labels obtained from denoising process: {clean_label_batch.shape}")