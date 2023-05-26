import torch
import torch.nn as nn

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

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch_size, 3, 64, 64]
        c1 = self.conv1(x) # (64x64x3) -> (31x31x32)
        h1 = self.relu(c1) # (31x31x32) -> (31x31x32)
        c2 = self.conv2(h1) # (31x31x32) -> (14x14x64)
        h2 = self.relu(c2) # (14x14x64) -> (14x14x64)
        c3 = self.conv3(h2) # (14x14x64) -> (6x6x128)
        h3 = self.relu(c3) # (6x6x128) -> (6x6x128)
        c4 = self.conv4(h3) # (6x6x128) -> (2x2x256)
        h4 = self.relu(c4) # (2x2x256) -> (2x2x256)

        d1 = h4.view(-1, 256*2*2) # (2x2x256) -> (1024)

        mu = self.mu(d1) # (1024) -> (32)
        logvar = self.logvar(d1) # (1024) -> (32)
        var = torch.exp(logvar)
        std = torch.sqrt(var)

        ep = torch.randn_like(std)

        z = mu + ep * std # (32) -> (32)

        return z, mu, logvar
    

class Decoder(nn.Module):
    """Decoder"""
    def __init__(self, z_dim=32):
        super().__init__()
        self.z_dim = z_dim

        self.l1 = nn.Linear(z_dim, 1024*1*1)

        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        # z: [batch_size, 32]
        d1 = self.l1(z) # (32) -> (1024)
        d1 = d1.view(-1, 1024, 1, 1) # (1024) -> (1x1x1024)

        dc1 = self.deconv1(d1) # (1x1x1024) -> (5x5x128)
        h1 = self.relu(dc1) # (5x5x128) -> (5x5x128)
        dc2 = self.deconv2(h1) # (5x5x128) -> (13x13x64)
        h2 = self.relu(dc2) # (13x13x64) -> (13x13x64)
        dc3 = self.deconv3(h2) # (13x13x64) -> (30x30x32)
        h3 = self.relu(dc3) # (30x30x32) -> (30x30x32)
        dc4 = self.deconv4(h3) # (30x30x32) -> (64x64x3)
        x = self.sigmoid(dc4) # (64x64x3) -> (64x64x3)

        return x
    

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