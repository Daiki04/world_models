import torch
import torch.nn as nn
from torch.nn import functional as F

"""
Task : Car 
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
VAE                                      [1, 3, 64, 64]            --
├─Encoder: 1-1                           [1, 32]                   --
│    └─Conv2d: 2-1                       [32, 31, 31]              1,568
│    └─ReLU: 2-2                         [32, 31, 31]              --
│    └─Conv2d: 2-3                       [64, 14, 14]              32,832
│    └─ReLU: 2-4                         [64, 14, 14]              --
│    └─Conv2d: 2-5                       [128, 6, 6]               131,200
│    └─ReLU: 2-6                         [128, 6, 6]               --
│    └─Conv2d: 2-7                       [256, 2, 2]               524,544
│    └─ReLU: 2-8                         [256, 2, 2]               --
│    └─Linear: 2-9                       [1, 32]                   32,800
│    └─Linear: 2-10                      [1, 32]                   32,800
├─Decoder: 1-2                           [1, 3, 64, 64]            --
│    └─Linear: 2-11                      [1, 1024]                 33,792
│    └─ConvTranspose2d: 2-12             [1, 128, 5, 5]            3,276,928
│    └─ReLU: 2-13                        [1, 128, 5, 5]            --
│    └─ConvTranspose2d: 2-14             [1, 64, 13, 13]           204,864
│    └─ReLU: 2-15                        [1, 64, 13, 13]           --
│    └─ConvTranspose2d: 2-16             [1, 32, 30, 30]           73,760
│    └─ReLU: 2-17                        [1, 32, 30, 30]           --
│    └─ConvTranspose2d: 2-18             [1, 3, 64, 64]            3,459
│    └─Sigmoid: 2-19                     [1, 3, 64, 64]            --
==========================================================================================
Total params: 4,348,547
Trainable params: 4,348,547
Non-trainable params: 0
Total mult-adds (M): 597.50
==========================================================================================
Input size (MB): 0.05
Forward/backward pass size (MB): 0.84
Params size (MB): 17.39
Estimated Total Size (MB): 18.28
==========================================================================================
"""

class Encoder(nn.Module):
    """Encoder"""

    def __init__(self, z_dim=32):
        super().__init__()
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2) 
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.mu = nn.Linear(256*2*2, z_dim)
        self.logvar = nn.Linear(256*2*2, z_dim)

    def forward(self, x):
        # x: [batch_size, 3, 64, 64]
        c1 = F.relu(self.conv1(x)) # (3x64x64) -> (32x31x31)
        c2 = F.relu(self.conv2(c1)) # (32x31x31) -> (64x14x14)
        c3 = F.relu(self.conv3(c2)) # (64x14x14) -> (128x6x6)
        c4 = F.relu(self.conv4(c3)) # (128x6x6) -> (256x2x2)

        # d1 = c4.view(-1, 256*2*2) # (256x2x2) -> (1024)
        d1 = c4.reshape(-1, 256*2*2) # (256x2x2) -> (1024)

        mu = self.mu(d1) # (1024) -> (32)
        logvar = self.logvar(d1) # (1024) -> (32)
        std = torch.exp(0.5 * logvar) # (32)
        ep = torch.randn_like(std) # (32)

        z = mu + ep * std # (32) -> (32)

        return z, mu, logvar
    

class Decoder(nn.Module):
    """Decoder"""
    def __init__(self, z_dim=32):
        super().__init__()
        self.z_dim = z_dim

        self.l1 = nn.Linear(z_dim, 1024)

        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)
    
    def forward(self, z):
        # z: [batch_size, 32]
        d1 = self.l1(z) # (32) -> (1024)
        # d1 = d1.view(-1, 1024, 1, 1) # (1024) -> (1024x1x1)
        d1 = d1.reshape(-1, 1024, 1, 1) # (1024) -> (1024x1x1)

        dc1 = F.relu(self.deconv1(d1)) # (1024x1x1) -> (128x5x5)
        dc2 = F.relu(self.deconv2(dc1)) # (128x5x5) -> (64x13x13)
        dc3 = F.relu(self.deconv3(dc2)) # (64x13x13) -> (32x30x30)
        dc4 = torch.sigmoid(self.deconv4(dc3)) # (32x30x30) -> (3x64x64)

        return dc4
    

class VAE(nn.Module):
    '''VAE'''
    def __init__(self, z_dim=32):
        super().__init__()
        self.z_dim = z_dim

        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        x = self.decoder(z)

        return x, mu, logvar
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
def loss_function(label, predict, mu, log_var, kl_tolerance=0.5, z_size=32):
    r_loss = torch.sum((predict - label).pow(2), dim=(1, 2, 3))
    r_loss = torch.mean(r_loss)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    kl_loss = torch.max(kl_loss, kl_loss.new([kl_tolerance * z_size]))
    kl_loss = torch.mean(kl_loss)
    return r_loss + kl_loss, r_loss, kl_loss