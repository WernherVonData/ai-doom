import utils

import torch.nn as nn

from katie.rl import SoftmaxBody


class Agent:

    def __init__(self, scenario_name, agent_identifier, temperature=1.0, image_dim=80, lr=0.001):
        self.scenario_path, self.nb_available_buttons = utils.get_path_and_number_of_actions_to_scenario(scenario_name=scenario_name)
        self.actions = []
        for i in range(0, self.nb_available_buttons):
            self.actions.append(
                [True if action_index == i else False for action_index in range(0, self.nb_available_buttons)])
        self.image_dim = image_dim
        self.softmax_body = SoftmaxBody(temperature=temperature)
        self.loss = nn.MSELoss()
        self.lr = lr
        self.agent_identifier = agent_identifier
        self.last_image = None
        self.last_reward = None
        self.cnn = None
        self.ai = None

    def read_state(self, state):
        raise NotImplementedError("Method {} must be implemented in the child class".format("read_state"))
        yield

    def make_action(self, state_data):
        raise NotImplementedError("Method {} must be implemented in the child class".format("make_action"))
        yield

    def calculate_reward(self, game_reward):
        raise NotImplementedError("Method {} must be implemented in the child class".format("calculate_reward"))
        yield

    def generate_history_record(self, action, game_finished):
        raise NotImplementedError("Method {} must be implemented in the child class".format("generate_history_record"))
        yield

    def perform_training_step(self, batch, gamma):
        raise NotImplementedError("Method {} must be implemented in the child class".format("perform_training_step"))
        yield

    def eligibility_trace(self, batch, gamma):
        raise NotImplementedError("Method {} must be implemented in the child class".format("eligibility_trace"))
        yield

    def load_agent_optimizer_and_replay_memory(self, model_path, memory_path):
        raise NotImplementedError("Method {} must be implemented in the child class".format("load_agent_optimizer_and_replay_memory"))
        yield
