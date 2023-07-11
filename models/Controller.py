import numpy as np

class Controller:
    """Controller"""
    def __init__(self, z_size, a_size, h_size, params):
        self.z_size = z_size
        self.a_size = a_size
        self.h_size = h_size

        self.input_size = z_size + h_size

        self.bias = np.array(params[:self.a_size])
        self.weight = np.array(params[self.a_size:]).reshape(self.input_size, self.a_size)

    def forward(self, z, h):
        input = np.concatenate((z, h), axis=0)
        # print(input.shape, self.weight.shape, self.bias.shape)
        a = np.dot(self.weight.T, input) + self.bias
        a = np.tanh(a)
        a[1] = (a[1] + 1.0) / 2.0
        a[2] = np.minimum(np.maximum(a[2], 0), 1.0)
        return a
    
    def forward_onlyz(self, z):
        input = z
        a = np.dot(self.weight.T, input) + self.bias
        a = np.tanh(a)
        a[1] = (a[1] + 1.0) / 2.0
        a[2] = np.minimum(np.maximum(a[2], 0), 1.0)
        return a

    
def get_random_params(z_size, a_size, h_size, sigma):
    """ランダムなパラメータを生成"""
    params_size = (z_size + h_size) * a_size + a_size
    return np.random.standard_cauchy(params_size) * sigma