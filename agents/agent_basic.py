from collections import namedtuple

import numpy as np
import torch

from agent import Agent
import cnn_agent
import image_preprocessing
import utils
import torch.optim as optim


_Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])


class AgentBasic(Agent):

    def __init__(self, scenario_name, agent_identifier, temperature=1.0, image_dim=80, lr=0.001):
        super.__init__(scenario_name, agent_identifier, temperature, image_dim, lr)
        self.cnn = cnn_agent.CNN(number_actions=self.nb_available_buttons, image_dim=image_dim)
        self.cnn.to(utils.DEVICE_NAME)
        self.ai = cnn_agent.AI(brain=self.cnn, body=self.softmax_body)
        self.optimizer = optim.Adam(self.cnn.parameters(), lr=self.lr)

    def read_state(self, state):
        buffer = state
        self.last_image = image_preprocessing.to_grayscale_and_resize(buffer, self.image_dim, self.image_dim)
        return np.array([self.last_image])

    def make_action(self, state_data):
        return self.ai(state_data)[0][0]

    def calculate_reward(self, game_reward):
        self.last_reward = game_reward
        return self.last_reward

    def generate_history_record(self, action, game_finished):
        return _Step(state=self.last_image, action=action, reward=self.last_reward, done=game_finished)

    def eligibility_trace(self, batch, gamma=0.99):
        inputs = []
        targets = []
        for series in batch:
            input = torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32))
            output = self.cnn(input.to(utils.DEVICE_NAME))
            cumul_reward = 0.0 if series[-1].done else output[1].data.max()
            for step in reversed(series[:-1]):
                cumul_reward = step.reward + gamma * cumul_reward
            state = series[0].state
            target = output[0].data
            target[series[0].action] = cumul_reward
            inputs.append(state)
            targets.append(target)
        return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)

    def load_agent_optimizer_and_replay_memory(self, model_path, memory_path):
        self.cnn, self.optimizer = utils.load(model_path, model_used=self.cnn, optimizer_used=self.optimizer)
        self.memory.load_memory_buffer(memory_path)