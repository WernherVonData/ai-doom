import numpy as np
import torch
from torch.autograd import Variable

from . import agent
from . import cnn_agent_second_input
from . import history_records
import image_preprocessing
import utils
import torch.optim as optim
import vizdoom as viz


class AgentTwoInput(agent.Agent):

    def __init__(self, scenario_name, agent_identifier, temperature=1.0, image_dim=80, lr=0.001):
        super(AgentTwoInput, self).__init__(scenario_name, agent_identifier, temperature, image_dim, lr)
        self.cnn = cnn_agent_second_input.CnnTwoInput(number_actions=self.nb_available_buttons, image_dim=image_dim, linear_input=4)
        self.cnn.to(utils.DEVICE_NAME)
        self.ai = cnn_agent_second_input.AITwoInput(brain=self.cnn, body=self.softmax_body)
        self.optimizer = optim.Adam(self.cnn.parameters(), lr=self.lr)
        self.current_health = 100
        self.step_nb = -1
        self.previous_health = 100
        self.linear_input = []
        self.delta_health = 0

    def read_game_data(self, game):
        state = game.get_state()
        buffer = state.screen_buffer
        self.last_image = self.screen_processing(buffer, self.image_dim, self.image_dim)
        self.current_health = game.get_game_variable(viz.GameVariable.HEALTH)
        step = state.number
        self.delta_health = 0
        if self.previous_health >= self.current_health:
            self.delta_health = abs(self.current_health - self.current_health)
            self.previous_health = self.current_health
        self.linear_input = np.array([[self.current_health, self.previous_health, self.delta_health, step]])
        return [np.array([self.last_image]), self.linear_input]

    def make_action(self, state_data):
        return self.ai(state_data[0], state_data[1])[0][0]

    def calculate_reward(self, game_reward):
        reward_to_save = 0
        reward_to_save -= self.delta_health
        if self.current_health > 0:
            reward_to_save += 10
        reward_to_save += game_reward

        self.last_reward = reward_to_save
        return self.last_reward

    def generate_history_record(self, action, game_finished):
        return history_records.LinearStep(state=self.last_image, linear=self.linear_input, action=action, reward=self.last_reward,
                                         done=game_finished)

    def perform_training_step(self, batch, gamma):
        image_inputs, linear_inputs, targets = self.eligibility_trace(batch=batch, gamma=gamma)
        image_inputs, linear_inputs, targets = Variable(image_inputs).to(utils.DEVICE_NAME), Variable(linear_inputs).to(
            utils.DEVICE_NAME), Variable(targets)
        predictions = self.cnn(image_inputs, linear_inputs)
        loss_error = self.loss(predictions, targets)
        self.optimizer.zero_grad()
        loss_error.backward()
        self.optimizer.step()

    def eligibility_trace(self, batch, gamma=0.99):
        inputs = []
        inputs_hp = []
        targets = []
        for series in batch:
            input = torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32))
            linear_input = torch.from_numpy(np.array([series[0].linear, series[-1].linear], dtype=np.float32))
            output = self.cnn(input.to(utils.DEVICE_NAME), linear_input.to(utils.DEVICE_NAME))
            cumul_reward = 0.0 if series[-1].done else output[1].data.max()
            for step in reversed(series[:-1]):
                cumul_reward = step.reward + gamma * cumul_reward
            state = series[0].state
            hp = series[0].linear
            target = output[0].data
            target[series[0].action] = cumul_reward
            inputs.append(state)
            inputs_hp.append(hp)
            targets.append(target)
        return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.from_numpy(np.array(inputs_hp, dtype=np.float32)), torch.stack(targets)

    def load_agent_optimizer(self, model_path):
        self.cnn, self.optimizer = utils.load(model_path, model_used=self.cnn, optimizer_used=self.optimizer)
