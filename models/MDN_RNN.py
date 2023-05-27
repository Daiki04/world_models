import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormBasicLSTM(nn.Module):
    """LSTM"""
    def __init__(self, z_size, a_size, hidden_size=256, num_layers=1):
        super().__init__()
        self.z_size = z_size
        self.a_size = a_size
        self.input_size = z_size + a_size
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=False) # LSTM：入力サイズ、隠れ層サイズ、層数、バイアス、バッチファースト

    def forward(self, x):
        # x: [seq_len, batch_size, z_size + a_size]
        output, (h_n, c_n) = self.lstm(x)
        # output: [seq_len, batch_size, hidden_size]
        # h_n: [num_layers, batch_size, hidden_size]
        # c_n: [num_layers, batch_size, hidden_size]
        return output, (h_n, c_n)


class MDN(nn.Module):
    """Mixture Density Network"""
    def __init__(self, hidden_size=256, z_size=32, r_size=0, num_gaussians=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.z_size = z_size
        self.num_gaussians = num_gaussians
        self.output_size = num_gaussians * 3 *  z_size + r_size # 3: mu, sigma, pi, r_size: reward

        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        # x: [seq_len, batch_size, hidden_size]
        output = self.fc(x)
        # output: [seq_len, batch_size, num_gaussians * 3 * z_size + r_size]
        p, mu, sigma, r = self.split_mdn_params(output)

        return p, mu, sigma, r
    
    def split_mdn_params(self, output):
        # output: [seq_len, batch_size, num_gaussians * 3 * z_size + r_size]
        p = output[:, :, :self.num_gaussians]
        mu = output[:, :, self.num_gaussians:self.num_gaussians * (self.z_size + 1)]
        sigma = output[:, :, self.num_gaussians * (self.z_size + 1):self.num_gaussians * (self.z_size * 2 + 1)]
        r = output[:, :, -1]

        # p: [seq_len, batch_size, num_gaussians]
        # mu: [seq_len, batch_size, num_gaussians * z_size]
        # sigma: [seq_len, batch_size, num_gaussians * z_size]
        # r: [seq_len, batch_size, r_size]

        p = F.softmax(p, dim=2)
        sigma = torch.exp(sigma)
        return p, mu, sigma, r
    

class MDRNN(nn.Module):
    """MDN-RNN"""
    def __init__(self, z_size, a_size, r_size, hidden_size=256, num_layers=1, num_gaussians=5):
        super().__init__()
        self.z_size = z_size
        self.a_size = a_size
        self.r_size = r_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_gaussians = num_gaussians

        self.lstm = LayerNormBasicLSTM(z_size, a_size, hidden_size, num_layers)
        self.mdn = MDN(hidden_size, z_size, r_size, num_gaussians)

    def forward(self, z, a):
        # z: [seq_len, batch_size, z_size]
        # a: [seq_len, batch_size, a_size]
        za = torch.cat([z, a], dim=2)
        # za: [seq_len, batch_size, z_size + a_size]
        output, (h_n, c_n) = self.lstm(za)
        # output: [seq_len, batch_size, hidden_size]
        # h_n: [num_layers, batch_size, hidden_size]
        # c_n: [num_layers, batch_size, hidden_size]
        p, mu, sigma, r = self.mdn(output)
        # p: [seq_len, batch_size, num_gaussians]
        # mu: [seq_len, batch_size, num_gaussians * z_size]
        # sigma: [seq_len, batch_size, num_gaussians * z_size]
        # r: [seq_len, batch_size, r_size]
        return output, p, mu, sigma, r
