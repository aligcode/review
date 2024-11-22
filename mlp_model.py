import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvTwoLayerMLP(nn.Module):
    
    def __init__(self, input_dim=3, num_classes=2, pretrain=False):
        super(ConvTwoLayerMLP, self).__init__()
        self.pretrain = pretrain
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128, 32, bias=True)
        self.fc2 = nn.Linear(32, num_classes)

        if self.pretrain:
            self.linear_l = nn.Linear(128, 32, bias=True)

    def encode(self, x):
        batch_size = x.shape[0]
        l1_out = self.relu(self.bn1(self.conv1(x)))
        l2_out = self.relu(self.bn2(self.conv2(l1_out)))
        img_feat = F.adaptive_avg_pool2d(l2_out, output_size=1)
        img_feat_flattened = img_feat.view(batch_size, -1)
        return img_feat_flattened
    
    def forward_pretrain(self, x):
        img_feat_flattened = self.encode(x)
        view_projection = self.linear_l(img_feat_flattened)
        return view_projection
    
    def forward_train(self, x):
        img_feat_flattened = self.encode(x)
        logits = self.fc2(self.relu(self.fc1(img_feat_flattened)))
        return logits
    
    def forward(self, x):
        if self.pretrain:
            return self.forward_pretrain(x)
        else:
            return self.forward_train(x)

def main():
    model = ConvTwoLayerMLP(input_dim=3, num_classes=2)
    batch_size, num_channels, height, width = 16, 3, 256, 256
    x = torch.randn(batch_size, num_channels, height, width)
    logits = model(x)
    print(f"Model outputs for x of size {x.shape}: {logits.shape}")
    
if __name__ == '__main__':
    main()
        