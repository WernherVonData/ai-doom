from collections import namedtuple

import numpy as np
import torch

import utils
from agent import Agent
import image_preprocessing

_Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])

class AgentBasic(Agent):


    def __init__(self):
        self.last_image = None
        self.last_reward = None

    def read_state(self, state, image_dim):
        buffer = state
        self.last_image = image_preprocessing.to_grayscale_and_resize(buffer, image_dim, image_dim)
        return np.array([self.last_image])

    def make_action(self, ai, state_data):
        return ai(state_data)[0][0]

    def calculate_reward(self, game_reward):
        self.last_reward = game_reward
        return self.last_reward

    def generate_history_record(self, action, game_finished):
        return _Step(state=self.last_image, action=action, reward=self.last_reward, done=game_finished)

    def eligibility_trace(self, cnn, batch, gamma=0.99):
        inputs = []
        targets = []
        for series in batch:
            input = torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32))
            output = cnn(input.to(utils.DEVICE_NAME))
            cumul_reward = 0.0 if series[-1].done else output[1].data.max()
            for step in reversed(series[:-1]):
                cumul_reward = step.reward + gamma * cumul_reward
            state = series[0].state
            target = output[0].data
            target[series[0].action] = cumul_reward
            inputs.append(state)
            targets.append(target)
        return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)
