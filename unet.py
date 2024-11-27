import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            dilation=1
        )
        self.bn = nn.BatchNorm2d(
            num_features=out_channels
        )
        self.relu = nn.ReLU(inplace=True)
      
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DeconvBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.deconv = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride
        )

    def forward(self, x):
        return self.deconv(x)

class VariationalDecoder(nn.Module):
    
    def __init__(self, num_hidden_channels, output_size, output_channels):
        super(VariationalDecoder, self).__init__()
        
        self.num_hidden_channels = num_hidden_channels
        self.output_size = output_size
        self.output_channels = output_channels

    
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=self.num_hidden_channels,  # 256
            out_channels=self.num_hidden_channels//2, # 128
            kernel_size=34, 
            stride=3
        )
        
        self.decode1 = ConvBlock( # 128 + 128
            in_channels=128+256, # self.num_hidden_channels//2 + self.num_hidden_channels//2 (skip connection)
            out_channels=128, # 64
            stride=1 # keep spatial dimension
        )
        self.bn1 = nn.BatchNorm2d(num_features=128)
        
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=128, # 64, 64
            out_channels=128, # 32
            kernel_size=16, 
            stride=3
        )
        
        self.decode2 = ConvBlock( # 32, 32
            in_channels=self.num_hidden_channels, # self.num_hidden_channels//2 + self.num_hidden_channels//2 (skip connection)
            out_channels=1, # 1
            stride=1 # keep spatial dimension
        )
        
        self.upsample_final = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=1,
            kernel_size=8, 
            stride=3
        )
        self.relu = nn.ReLU()
        
    def forward(self, enc2, enc3, bottleneck):
        bottleneck = bottleneck.unsqueeze(-1).unsqueeze(-1)
        upsampled_1 = self.deconv1(bottleneck)

        merged_1 = torch.cat([upsampled_1, enc3], dim=1)
        dec1 = self.decode1(merged_1)
        dec1 = self.relu(self.bn1(dec1))
        
        upsampled_2 = self.deconv2(dec1) 
        diff = enc2.shape[2] - upsampled_2.shape[2] 
        pad = diff//2
        padding = (pad, pad+1, pad+1, pad) # left, right, top, bottom
        upsampled_2_padded = F.pad(upsampled_2, padding, mode='constant', value=0)
        merged_2 = torch.cat([upsampled_2_padded, enc2], dim=1)
        dec2 = self.decode2(merged_2)
        upsampled_final = self.upsample_final(dec2)
        padding = (2, 1, 2, 1)
        upsampled_final_padded = F.pad(upsampled_final, padding, mode='constant', value=0)
        return upsampled_final_padded

class VariationalEncoder(nn.Module):
    
    def __init__(self, num_input_channels, num_hidden_channels):
        super(VariationalEncoder, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_hidden_channels = num_hidden_channels
        
        assert self.num_hidden_channels > 128
        
        self.conv1 = ConvBlock(
            in_channels=self.num_input_channels, 
            out_channels=32,
            stride=1
        )
        
        self.conv2 = ConvBlock(
            in_channels=32,
            out_channels=128,
            stride=3
        )
        
        self.conv3 = ConvBlock(
            in_channels=128,
            out_channels=self.num_hidden_channels,
            stride=5
        )
        
        self.conv_final = nn.Conv2d(
            in_channels=self.num_hidden_channels,
            out_channels=self.num_hidden_channels,
            kernel_size=1,
            stride=1,
            dilation=1
        )
        
        self.bn = nn.BatchNorm1d(num_features=self.num_hidden_channels)
        self.relu = nn.ReLU()
        self.mean = nn.Linear(self.num_hidden_channels, self.num_hidden_channels)
        self.logvar = nn.Linear(self.num_hidden_channels, self.num_hidden_channels)
        
    def encode(self, x):
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc_last = F.adaptive_avg_pool2d(self.conv_final(enc3), output_size=1)
        
        return {
            'enc1': enc1,
            'enc2': enc2, 
            'enc3': enc3,
            'enc_last': enc_last
        }
        
    def forward(self, x):
        # x: input image [batch, height, width, 1]
        encoded = self.encode(x)
        encoded_bn_relud = self.relu(self.bn(encoded['enc_last'].squeeze()))
        encoded.update({
            'mean': self.mean(encoded_bn_relud),
            'logvar': self.logvar(encoded_bn_relud)
        })
        
        return encoded
    
    
class VariationalUNet(nn.Module):
    
    def __init__(self):
        super(VariationalUNet, self).__init__()
        
        self.encoder = VariationalEncoder(num_input_channels=1, num_hidden_channels=256)
        self.decoder = VariationalDecoder(num_hidden_channels=256, output_size=512, output_channels=1)
        
    def forward(self, x):
        
        res = self.encoder(x)
        mean = res['mean'] # batch, 256
        logvar = res['logvar'] # batch, 256
        enc_last = res['enc_last'] # batch, 256
        enc2 = res['enc2']
        enc3 = res['enc3']
    
        batch_size, hidden_dim = mean.shape[0], mean.shape[1]
        
        # reparam
        sample = mean + torch.exp(logvar) * torch.randn(batch_size, hidden_dim)
        x_hat = self.decoder(enc2, enc3, sample)
        
        return x_hat
    
if __name__ == '__main__':
    
    conv_block = ConvBlock(in_channels=1, out_channels=32, stride=3)
    variational_encoder = VariationalEncoder(num_input_channels=1, num_hidden_channels=256)
    batch_size = 16
    height = 512
    width = 512
    num_channels = 1
    x = torch.randn(batch_size, num_channels, height, width)
    num_hidden_channels = 256
    variational_decoder = VariationalDecoder(num_hidden_channels=num_hidden_channels, output_size=512, output_channels=1)
    
    print(f"x: {x.shape}")
    print(f"output of conv block {conv_block(x).shape}")
    
    res = variational_encoder(x)
    encoded, mean, var = res['enc_last'], res['mean'], res['logvar']
    print(f"encoding: {encoded.shape} | mean: {mean.shape} | logvar {var.shape}")
    
    enc1, enc2, enc3 = res['enc1'], res['enc2'], res['enc3']
    print(f"enc1: {enc1.shape} | enc2: {enc2.shape} | enc3: {enc3.shape}")
    
    sample_x = torch.randn(batch_size, num_hidden_channels)
    x_hat = variational_decoder(enc2, enc3, sample_x)
    print(f"x.shape {x.shape} | x_hat.shape {x_hat.shape}")
    
    variational_unet = VariationalUNet()
    print("variational UNet x_hat: ", variational_unet(x).shape)