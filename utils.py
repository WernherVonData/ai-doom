import torch
import numpy as np
import os

DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'


class MemoryAverage:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size

    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]

    def average(self):
        return np.mean(self.list_of_rewards)


def save(filename, model, optimizer):
    torch.save({'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, filename)


def load(agent_file, model_used, optimizer_used):
    if os.path.isfile(agent_file):
        print("=>loading agent")
        agent = torch.load(agent_file)
        model_used.load_state_dict(agent['state_dict'])
        optimizer_used.load_state_dict(agent['optimizer'])
    else:
        print("no checkpoint found...")
    return model_used, optimizer_used


def get_path_and_number_of_actions_to_scenario(scenario_name):
    if scenario_name == "basic":
        return "scenarios/basic.cfg", 3
    if scenario_name == "rocket":
        return "scenarios/rocket_basic.cfg", 3
    if scenario_name == "corridor":
        return "scenarios/deadly_corridor.cfg", 7
    print("=> Scenario {} has wrong name or it's not supported yet".format(scenario_name))
    return None, None
