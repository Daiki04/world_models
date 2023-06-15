import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    """LSTM"""
    def __init__(self, z_size=32, a_size=3, hidden_size=256, num_layers=1):
        super().__init__()
        self.z_size = z_size
        self.a_size = a_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(z_size + a_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, z, a, hidden, cell):
        input = torch.cat((z, a), dim=2)
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        return output, hidden, cell
    
class MDN(nn.Module):
    """Mixtures Density Network"""
    def __init__(self, z_size=32, hidden_size=256, num_layers=1, num_mixtures=5, use_reward=False, use_Done=False):
        super().__init__()
        self.z_size = z_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        self.use_reward = use_reward
        self.use_Done = use_Done

        if use_reward and use_Done:
            self.output_size = 3 * num_mixtures * z_size + 2
        elif use_reward or use_Done:
            self.output_size = 3 * num_mixtures * z_size + 1
        else:
            self.output_size = 3 * num_mixtures * z_size
        
        self.fc = nn.Linear(hidden_size, self.output_size)
    
    def forward(self, x):
        output = self.fc(x)
        # return output
        if self.use_reward and self.use_Done:
            pi, mu, logsigma, reward, done = self.get_mixture(output)
            return pi, mu, logsigma, reward, done
        elif self.use_reward or self.use_Done:
            pi, mu, logsigma, reward_done = self.get_mixture(output)
            return pi, mu, logsigma, reward_done
        else:
            pi, mu, logsigma = self.get_mixture(output)
            return pi, mu, logsigma
    
    def get_mixture(self, output):
        params_dim = self.num_mixtures * self.z_size
        pi = output[:, :, :params_dim]
        mu = output[:, :, params_dim:2*params_dim]
        logsigma = output[:, :, 2*params_dim:3*params_dim]

        if self.use_reward and self.use_Done:
            reward = output[:, :, 3*params_dim]
            done = output[:, :, 3*params_dim+1]
            return pi, mu, logsigma, reward, done
        elif self.use_reward or self.use_Done:
            reward_done = output[:, :, 3*params_dim]
            return pi, mu, logsigma, reward_done
        else:
            return pi, mu, logsigma
        
class MDNRNN(nn.Module):
    """MDN-RNN"""
    def __init__(self, z_size=32, a_size=3, hidden_size=256, num_layers=1, num_mixtures=5, use_reward=False, use_Done=False):
        super().__init__()
        self.z_size = z_size
        self.a_size = a_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        self.use_reward = use_reward
        self.use_Done = use_Done

        self.lstm = LSTM(z_size, a_size, hidden_size, num_layers)
        self.mdn = MDN(z_size, hidden_size, num_layers, num_mixtures, use_reward, use_Done)
    
    def forward(self, z, a, hidden, cell):
        output, hidden, cell = self.lstm(z, a, hidden, cell)
        pi, mu, logsigma = self.mdn(output)
        # mdn_output = self.mdn(output)
        return pi, mu, logsigma, hidden, cell
        # return mdn_output, hidden, cell

def loss_func(y_true, pi, mu, logsigma):
    """MDN Loss Function
    負の対数尤度
    pi: [seq_len, batch_size, num_gaussians*z_size]
    mu: [seq_len, batch_size, num_gaussians*z_size]
    logsigma: [seq_len, batch_size, num_gaussians*z_size]
    y_true: [seq_len, batch_size, z_size]
    """
    # pi: [seq_len, batch_size, num_gaussians*z_size] -> [seq_len, batch_size, num_gaussians, z_size]
    pi = pi.view(pi.size(0), pi.size(1), -1, 32)
    log_softmax_pi = F.log_softmax(pi, dim=2)
    # mu: [seq_len, batch_size, num_gaussians*z_size] -> [seq_len, batch_size, num_gaussians, z_size]
    mu = mu.view(mu.size(0), mu.size(1), -1, 32)
    # logsigma: [seq_len, batch_size, num_gaussians*z_size] -> [seq_len, batch_size, num_gaussians, z_size]
    logsigma = logsigma.view(logsigma.size(0), logsigma.size(1), -1, 32)

    # print("log_softmax_pi.shape: ", log_softmax_pi.shape)
    # print("mu.shape: ", mu.shape)
    # print("logsigma.shape: ", logsigma.shape)
    # print("y_true.shape: ", y_true.shape)
    log_gauss = -0.5 * (2*logsigma + math.log(2 * math.pi) + (y_true.unsqueeze(2) - mu)**2 / (2 * torch.exp(logsigma)**2))

    loss = -torch.logsumexp(log_softmax_pi + log_gauss, dim=2)
    return loss.mean()
