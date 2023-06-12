import tqdm
import os
import gym
from PIL import Image
import numpy as np
import numpy.random as nr
import argparse
from models.VAE import VAE
import torch

class CarRacing_rollouts():
    def __init__(self, seed_num=0):
        self.env = gym.make('CarRacing-v2', render_mode='rgb_array', domain_randomize=True)
        self.env.reset(seed=seed_num)
        self.file_dir = './data/CarRacing/'

    def get_rollouts(self, num_rollouts=10000, reflesh_rate=20, max_episode=3000):
        start_idx = 0
        if os.path.exists(self.file_dir):
            start_idx = len(os.listdir(self.file_dir)) 
        for i in tqdm.tqdm(range(start_idx, num_rollouts+1)):
            state_sequence = []
            action_sequence = []
            reward_sequence = []
            done_sequence = []
            state = self.env.reset()
            done = False
            iter = 0
            while (not done) and iter < max_episode:
                if iter % reflesh_rate == 0:
                    if iter < 20:
                        steering = -0.1
                        acceleration = 1
                        brake = 0
                    else:
                        steering = nr.uniform(-1, 1)
                        acceleration = nr.uniform(0, 1)
                        brake = nr.uniform(0, 1)
                action = np.array([steering, acceleration, brake])
                state, reward, done, _, _ = self.env.step(action)
                state = self.reshape_state(state)
                state_sequence.append(state)
                action_sequence.append(action)
                reward_sequence.append(reward)
                done_sequence.append(done)
                iter += 1
            np.savez_compressed(os.path.join(self.file_dir, 'rollout_{}.npz'.format(i)), state=state_sequence, action=action_sequence, reward=reward_sequence, done=done_sequence)
            # np.savez(os.path.join(self.file_dir, 'rollout_{}.npz'.format(i)), state=state_sequence, action=action_sequence, reward=reward_sequence, done=done_sequence)

    def load_rollout(self, idx_rolloout):
        data = np.load(os.path.join(self.file_dir, 'rollout_{}.npz'.format(idx_rolloout)))
        return data['state'], data['action'], data['reward'], data['done']
    
    def rollout_to_z(self):
        vae = VAE()
        checkpoint = torch.load('./vae.pth')
        vae.load_state_dict(checkpoint['model_state_dict'])
        vae.eval()

        with torch.no_grad():
            for i in tqdm.tqdm(range(10001)):
                state, action, reward, done = self.load_rollout(i)
                state = torch.tensor(state).float()
                state = state.permute(0, 3, 1, 2)
                z, mu, logvar = vae.encode(state)
                z, mu, logvar = z.detach().numpy(), mu.detach().numpy(), logvar.detach().numpy()
                np.savez_compressed("./data_z/rollout_z_{}.npz".format(i), z=z, mu=mu, logvar=logvar, action=action, reward=reward, done=done)

    def reshape_state(self, state):
        HEIGHT = 64
        WIDTH = 64
        state = state[0:84, :, :]
        state = np.array(Image.fromarray(state).resize((HEIGHT, WIDTH)))
        state = state / 255.0
        return state
    
    def make_gif(self, idx_rolloout):
        state, _, _, _ = self.load_rollout(idx_rolloout, self.file_dir)
        state = state * 255.0
        state = state.astype(np.uint8)
        for i in range(len(state)):
            img = Image.fromarray(state[i])
            img.save(os.path.join(self.file_dir, 'rollout_{}.gif'.format(idx_rolloout, i)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_rollouts', type=int, default=10000)
    parser.add_argument('--max_episode', type=int, default=3000)
    parser.add_argument('--reflesh_rate', type=int, default=20)
    parser.add_argument('--env_name', type=str, default='CarRacing')
    args = parser.parse_args()
    if args.env_name == 'CarRacing':
        env = CarRacing_rollouts()
    else:
        raise NotImplementedError
    
    env.get_rollouts(args.num_rollouts, args.reflesh_rate, args.max_episode)