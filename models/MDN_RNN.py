import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    """LSTM"""
    def __init__(self, z_size=32, a_size=3, hidden_size=256):
        super().__init__()
        self.z_size = z_size
        self.a_size = a_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(z_size + a_size, hidden_size)
    
    def forward(self, z, a, h, c):
        input = torch.cat((z, a), dim=2) # (seq_len, batch, input_size)
        out, (h, c) = self.lstm(input, (h, c))
        return out, h, c
    
class MDN(nn.Module):
    """Mixtures Density Network"""
    def __init__(self, z_size=32, hidden_size=256,num_mixtures=5, use_reward=False, use_Done=False):
        super().__init__()
        self.z_size = z_size
        self.hidden_size = hidden_size
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
    def __init__(self, z_size=32, a_size=3, hidden_size=256, num_mixtures=5, use_reward=False, use_Done=False):
        super().__init__()
        self.z_size = z_size
        self.a_size = a_size
        self.hidden_size = hidden_size
        self.num_mixtures = num_mixtures
        self.use_reward = use_reward
        self.use_Done = use_Done

        self.lstm = LSTM(z_size, a_size, hidden_size)
        self.mdn = MDN(z_size, hidden_size, num_mixtures, use_reward, use_Done)
    
    def forward(self, z, a, hidden, cell):
        out, h, c = self.lstm(z, a, hidden, cell)
        pi, mu, logsigma = self.mdn(out)
        return pi, mu, logsigma, h, c

"""
input:
    pi: (seq_len, batch, num_mixtures * z_size)
    mu: (seq_len, batch, num_mixtures * z_size)
    logsigma: (seq_len, batch, num_mixtures * z_size)
    y_true: (seq_len, batch, z_size)
output:
    loss: mean((seq_len, batch, 1))
"""
import math

# def loss_func(seq_len, batch_size, z_size, num_mixtures, pi, mu, logsigma, y_true):
#     y_true = y_true.view(seq_len, batch_size, z_size, 1) # (seq_len, batch, 1, z_size)

#     pi = pi.view(seq_len, batch_size, z_size, num_mixtures) # (seq_len, batch, z_size, num_mixtures)
#     mu = mu.view(seq_len, batch_size, z_size, num_mixtures) # (seq_len, batch, z_size, num_mixtures)
#     logsigma = logsigma.view(seq_len, batch_size, z_size, num_mixtures) # (seq_len, batch, z_size, num_mixtures)

#     pi = pi - torch.max(pi, dim=3)[0].view(seq_len, batch_size, z_size, 1) # (seq_len, batch, z_size, num_mixtures)
#     logpi = nn.LogSoftmax(dim=3)(pi) # (seq_len, batch, z_size, num_mixtures)
#     loggausian = -0.5 * (2*logsigma + (y_true - mu)**2 / (torch.exp(logsigma))**2) # (seq_len, batch, z_size, num_mixtures)

#     loss = logpi + loggausian # (seq_len, batch, z_size, num_mixtures)
#     loss = torch.sum(loss, dim=3) # (seq_len, batch, num_mixtures)
#     loss = -loss # (seq_len, batch, num_mixtures)
#     return torch.mean(loss)

def loss_func(seq_len, batch_size, z_size, num_mixtures, pi, mu, logsigma, y_true):
    y_true = y_true.view(seq_len, batch_size, z_size, 1) # (seq_len, batch, z_size, 1)

    pi = pi.view(seq_len, batch_size, z_size, num_mixtures) # (seq_len, batch, z_size, num_mixtures)
    mu = mu.view(seq_len, batch_size, z_size, num_mixtures) # (seq_len, batch, z_size, num_mixtures)
    logsigma = logsigma.view(seq_len, batch_size, z_size, num_mixtures) # (seq_len, batch, z_size, num_mixtures)

    pi = pi - torch.max(pi, dim=3)[0].view(seq_len, batch_size, z_size, 1) # (seq_len, batch, z_size, num_mixtures)
    exppi = torch.exp(pi) # (seq_len, batch, z_size, num_mixtures)
    sumexppi = torch.sum(exppi, dim=3).view(seq_len, batch_size, z_size, 1) # (seq_len, batch, z_size, 1)
    logpi = pi - torch.log(sumexppi) # (seq_len, batch, z_size, num_mixtures)
    loggausian = -0.5 * (2*logsigma + (y_true - mu)**2 / (torch.exp(logsigma))**2) # (seq_len, batch, z_size, num_mixtures)

    loss = logpi + loggausian # (seq_len, batch, z_size, num_mixtures)
    loss = torch.exp(loss) # (seq_len, batch, z_size, num_mixtures)
    loss = torch.sum(loss, dim=3) # (seq_len, batch, z_size)
    loss = -torch.log(loss) # (seq_len, batch, z_size)
    loss = torch.sum(loss, dim=2) # (seq_len, batch)
    return torch.mean(loss)
