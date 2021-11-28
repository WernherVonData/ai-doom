import numpy as np
import torch
from torch.autograd import Variable

from . import agent
from . import cnn_agent
from . import history_records
import image_preprocessing
import utils
import torch.optim as optim


class AgentBasic(agent.Agent):

    def __init__(self, scenario_name, agent_identifier, temperature=1.0, image_dim=80, lr=0.001):
        super(AgentBasic, self).__init__(scenario_name, agent_identifier, temperature, image_dim, lr)
        self.cnn = cnn_agent.CNN(number_actions=self.nb_available_buttons, image_dim=image_dim)
        self.cnn.to(utils.DEVICE_NAME)
        self.ai = cnn_agent.AI(brain=self.cnn, body=self.softmax_body)
        self.optimizer = optim.Adam(self.cnn.parameters(), lr=self.lr)

    def read_game_data(self, game):
        buffer = game.get_state().screen_buffer
        self.last_image = image_preprocessing.to_grayscale_and_resize(buffer, self.image_dim, self.image_dim)
        return np.array([self.last_image])

    def make_action(self, state_data):
        return self.ai(state_data)[0][0]

    def calculate_reward(self, game_reward):
        self.last_reward = game_reward
        return self.last_reward

    def generate_history_record(self, action, game_finished):
        return history_records.BasicStep(state=self.last_image, action=action, reward=self.last_reward,
                                         done=game_finished)

    def perform_training_step(self, batch, gamma):
        image_inputs, targets = self.eligibility_trace(batch=batch, gamma=gamma)
        image_inputs, targets = Variable(image_inputs).to(utils.DEVICE_NAME), Variable(targets)
        predictions = self.cnn(image_inputs)
        loss_error = self.loss(predictions, targets)
        self.optimizer.zero_grad()
        loss_error.backward()
        self.optimizer.step()

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

    def load_agent_optimizer(self, model_path):
        self.cnn, self.optimizer = utils.load(model_path, model_used=self.cnn, optimizer_used=self.optimizer)
