# forward diffusion
import cv2
import numpy as np
import torch
import os

# img : original image
def forward_diffusion(img, t, alphas_cumprod):
    
    current_alpha_sqrt = torch.sqrt(alphas_cumprod[t])
    noise = torch.randn_like(img) # normal dist. mean 0, identity variance
    noisy_img = current_alpha_sqrt * img + torch.sqrt(1 - alphas_cumprod[t]) * noise
    return noisy_img
    


if __name__ == "__main__":
    
    img = cv2.imread('image.jpg') / 255.0 # bgr
    diffusion_output = 'diffusion/'
    os.makedirs(diffusion_output, exist_ok=True)
    
    total_forward_steps = 1000
    beta_start, beta_end = 0.0000001, 0.000002
    betas = torch.linspace(beta_start, beta_end, total_forward_steps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    imgs = {}
    
    for t in range(total_forward_steps):
        noise_img = forward_diffusion(torch.tensor(img, dtype=torch.float32), t=t, alphas_cumprod=alphas_cumprod)
        if t % 100 == 0:
            imgs[t] = noise_img
            
    for t, img in imgs.items():
        img_path = os.path.join(diffusion_output, f't_{t}.jpg')
        cv2.imwrite(img_path, (img.numpy() * 255).astype(np.uint8))
    
    print("Forward diffusion complete.")
    





