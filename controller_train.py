import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gym
import datetime
import time
import os
import tqdm
from multiprocessing import Process
# import cma
from PIL import Image
import torch.multiprocessing as mp
import warnings

warnings.simplefilter('ignore')
import optuna 
from gym.envs.box2d.car_dynamics import Car

from models.VAE import VAE
from models.MDN_RNN import MDNRNN
from models.controller import Controller, get_random_params

max_cpu = os.cpu_count()

z_size = 32
a_size = 3
h_size = 256

params_size = (z_size + h_size) * a_size + a_size

rewards_through_gens = []
generation = 1

def reshape_state(state):
    HEIGHT = 64
    WIDTH = 64
    state = state[0:84, :, :]
    state = np.array(Image.fromarray(state).resize((HEIGHT, WIDTH)))
    state = state / 255.0
    return state

def decide_action(v, m, c, state, hidden, cell, device, use_world_model=False):
    hidden = hidden.reshape(-1)
    cell = cell.reshape(-1)

    state = reshape_state(state)
    state = np.moveaxis(state, 2, 0)
    state = np.reshape(state, (-1, 3, 64, 64))
    state = torch.tensor(state, dtype=torch.float32).to(device)
    z, _, _ = v.encode(state)
    z = z.detach().numpy()
    z = z.reshape(-1)
    # a = c.forward_onlyz(z)
    a = c.forward(z, hidden)
    z = torch.tensor(z, dtype=torch.float32).to(device).reshape(1, 1, -1)
    a = torch.tensor(a, dtype=torch.float32).to(device).reshape(1, 1, -1)
    hidden = torch.tensor(hidden, dtype=torch.float32).to(device).reshape(1, 1, -1)
    cell = torch.tensor(cell, dtype=torch.float32).to(device).reshape(1, 1, -1)
    pi, mu, logsigma, next_hidden, next_cell = m(z, a, hidden, cell)

    pi, mu, logsigma = pi.detach().numpy().reshape(-1), mu.detach().numpy().reshape(-1), logsigma.detach().numpy().reshape(-1)
    next_hidden, next_cell = next_hidden.detach().numpy().reshape(-1), next_cell.detach().numpy().reshape(-1)
    a = a.detach().numpy().reshape(-1)
    if use_world_model:
        return a, pi, mu, logsigma, next_hidden, next_cell
    else:
        return a, next_hidden, next_cell

def play(params, seed_num=0):
    with torch.no_grad():
        device = torch.device('cpu')
        vae = VAE()
        checkpoint = torch.load('vae.pth')
        vae.load_state_dict(checkpoint['model_state_dict'])
        vae = vae.eval()
        vae.to(device)

        mdnrnn = MDNRNN().to(device)
        checkpoint = torch.load('mdnrnn.pth')
        mdnrnn.load_state_dict(checkpoint['model_state_dict'])
        mdnrnn = mdnrnn.eval()
        mdnrnn.to(device)

        controller = Controller(z_size, a_size, h_size, params)
        # controller = Controller(z_size, a_size, 0, params)

        env = gym.make('CarRacing-v2', render_mode='rgb_array', domain_randomize=True)
        env.reset(seed=seed_num)

        _NUM_TRIALS = 3
        agent_reward = 0

        for trial in range(_NUM_TRIALS):
            state, _ = env.reset()
            np.random.seed(int(str(time.time()*1000000)[10:13]))
            position = np.random.randint(len(env.track))
            env.car = Car(env.world, *env.track[position][1:4])

            hidden = torch.zeros(1, 1, 256).float().to(device)
            cell = torch.zeros(1, 1, 256).float().to(device)

            total_reward = 0.0
            steps = 0

            while True:
                action, hidden, cell = decide_action(vae, mdnrnn, controller, state, hidden, cell, device)
                state, reward, done, _, _ = env.step(action)
                total_reward += reward
                steps += 1
                if steps > 3000 or done:
                    break
        
            agent_reward += total_reward
        
        env.close()
        return -(agent_reward / _NUM_TRIALS)
    
def objective(trial):
    params_size = (z_size + h_size) * a_size + a_size
    params = np.random.standard_cauchy(params_size) * 0.1
    for i in range(params_size):
        params[i] = trial.suggest_uniform('param_{}'.format(i), -2.0, 2.0)
    return play(params)

def optimize(study_name, storage, n_trials):
    sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)

if __name__ == '__main__':
    DATABASE_URI = 'sqlite:///controller.db'
    study_name = 'controller_params1'

    n_trials = 2000
    now = 0

    n_trials = n_trials - now
    concurrency = 5 
    # max_cpuより使うCPUの数が多くないことを確認
    assert concurrency <= max_cpu
    n_trials_per_cpu = n_trials / concurrency

    # valueのみをログに表示
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 並列化
    workers = [Process(target=optimize, args=(study_name, DATABASE_URI, n_trials_per_cpu)) for _ in range(concurrency)]
    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()