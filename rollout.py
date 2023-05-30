import tqdm
import os
import gym
from PIL import Image
import numpy as np
import numpy.random as nr

class CarRacing_rollouts():
    def __init__(self):
        self.env = gym.make('CarRacing-v2', render_mode='rgb_array')
        self.file_dir = './data/CarRacing/'

    def get_rollouts(self, num_rollouts=10000, reflesh_rate=20):
        for i in tqdm.tqdm(range(num_rollouts)):
            state_sequence = []
            action_sequence = []
            reward_sequence = []
            done_sequence = []
            state = self.env.reset()
            done = False
            iter = 0
            while not done:
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

    def load_rollout(self, idx_rolloout):
        data = np.load(os.path.join(self.file_dir, 'rollout_{}.npz'.format(idx_rolloout)))
        return data['state'], data['action'], data['reward'], data['done']

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