import torch
import torch.nn as nn

class Controller(nn.Module):
    """Controller"""

    def __init__(self, a_size, h_size=256, z_size=32):
        super().__init__()
        self.a_size = a_size
        self.h_size = h_size
        self.z_size = z_size
        self.input_size = z_size + h_size
        self.fc = nn.Linear(self.input_size, a_size)

    def forward(self, z, h):
        # z: [batch_size, z_size]
        # h: [batch_size, h_size]
        zh = torch.cat([z, h], dim=1)
        # zh: [batch_size, z_size + h_size]
        a = self.fc(zh)
        # a: [batch_size, a_size]
        return a