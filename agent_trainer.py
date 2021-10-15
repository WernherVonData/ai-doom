from collections import deque, namedtuple
import datetime
from random import choice

import torch.nn as nn
import torch.optim as optim

import vizdoom as viz

import utils
from agents import cnn_agent, cnn_agent_second_input
from katie.rl import SoftmaxBody
from katie.rl import ReplayMemory
from utils import MemoryAverage


class AgentTrainer:
    def __init__(self, agent_name, scenario_name, temperature=1.0, memory_capacity=100000, lr=0.001):
        self.scenario_path, self.nb_available_buttons = utils.get_path_and_number_of_actions_to_scenario(scenario_name=scenario_name)
        self.actions = []
        for i in range(0, self.nb_available_buttons):
            self.actions.append([True if action_index == i else False for action_index in range(0, self.nb_available_buttons)])

        image_dim = 80
        self.cnn = None
        self.ai = None
        self.softmax_body = SoftmaxBody(temperature=temperature)
        if agent_name == "cnn_image":
            self.cnn = cnn_agent.CNN(number_actions=self.nb_available_buttons, image_dim=image_dim)
            self.ai = cnn_agent.AI(brain=self.cnn, body=self.softmax_body)
        if agent_name == "cnn_image_linear":
            self.cnn = cnn_agent_second_input.CnnTwoInput(number_actions=self.nb_available_buttons, image_dim=image_dim, linear_input=4)

        if self.cnn is None or self.ai is None:
            raise ValueError("Agent not instantiated properly")

        self.cnn.to(utils.DEVICE_NAME)
        self.ma = MemoryAverage(100)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.cnn.parameters(), lr=lr)
        self.memory = ReplayMemory(capacity=memory_capacity)

    def load_agent_optimizer_and_replay_memory(self, model_path, memory_path):
        self.cnn, self.optimizer = utils.load(model_path, model_used=self.cnn, optimizer_used=self.optimizer)
        self.memory.load_memory_buffer(memory_path)


    def train(self, step_reader, make_action, calculate_reward, generate_history, starting_epoch=1, batch_size=128, n_step=10, nb_steps=1000):
        """

        :param step_reader: must read state from the game input and return data in a form of the named tuple,
        so make_action can read it.
        :param make_action: receives the results of the step_reader and this object ai object and must return action
        :param calculate_reward: receives the reward from the game and based on that it can return own reward
        :param generate_history: generates the named tuple for the replay memory, should accept the action and information whether game is finished
        :param starting_epoch:
        :param batch_size:
        :return:
        """
        game = viz.DoomGame()
        game.load_config(self.scenario_path)
        game.init()
        game.new_episode()

        while True:
            if game.is_episode_finished():
                game.new_episode()
            histories = []
            history = deque()
            reward = 0.0
            while True:
                state = game.get_state()
                state_data = step_reader(state)
                action = make_action(self.ai, state_data) if self.memory.is_buffer_full() else choice(range(0, self.nb_available_buttons))
                game_reward = game.make_action(self.actions[action])
                reward += calculate_reward(game_reward)
                history.append(generate_history(action, game.is_episode_finished()))
                if len(history) > n_step + 1:
                    history.popleft()
                if len(history) == n_step + 1:
                    histories.append(tuple(history))
                if game.is_episode_finished():
                    if len(history) > n_step + 1:
                        history.popleft()
                    while len(history) >= 1:
                        histories.append(tuple(history))
                        history.popleft()
                    rewards.append(reward)
                    reward = 0.0
                    game.new_episode()
                    history.clear()
                    previous_hp = 100
                if len(histories) >= nb_steps:
                    break
                # TODO: End of that TODO
            for history in histories:
                self.memory.append_memory(history)
            if not self.memory.is_buffer_full():
                print("Memory not full {} of {}".format(self.memory.get_current_buffer_size(), self.memory.get_capacity()))
                continue
            start = datetime.datetime.now()
            for batch in self.memory.sample_batch(batch_size):
                image_inputs, linear_inputs, targets = eligibility_trace(cnn=cnn, batch=batch, gamma=gamma)
                image_inputs, linear_inputs, targets = Variable(image_inputs).to(utils.DEVICE_NAME), Variable(
                    linear_inputs).to(utils.DEVICE_NAME), Variable(targets)
                predictions = cnn(image_inputs, linear_inputs)
                loss_error = loss(predictions, targets)
                optimizer.zero_grad()
                loss_error.backward()
                optimizer.step()
            stop = datetime.datetime.now()
            delta = stop - start
            rewards_steps = rewards
            history_reward = history_reward + rewards
            rewards = []
            ma.add(rewards_steps)
            avg_reward = ma.average()
            avg_history_reward.append(avg_reward)
            print("Epoch: %s, Average Reward: %s Delta: %f Time: %s" % (
            str(epoch), str(avg_reward), delta.seconds, str(datetime.datetime.now())))
            epoch += 1
            if epoch % 10 == 0:
                model_file = "results\\cnn_doom_" + str(epoch) + ".pth"
                score_file = "results\\scores_" + str(epoch) + ".png"
                avg_score_file = "results\\avg_scores_" + str(epoch) + ".png"
                memory_file = "results\\buffer_" + str(epoch) + ".pickle"
                memory.save_memory_buffer(memory_file)
                print("Saving model file: {} and diagram: {}".format(model_file, score_file))
                plt.clf()
                plt.plot(history_reward, color='blue')
                plt.savefig(score_file)
                plt.clf()
                plt.plot(avg_history_reward, color='green')
                plt.savefig(avg_score_file)
                utils.save(model_file, cnn, optimizer)
            if epoch == nb_epochs:
                print("Reached last epoch, finishing...")
                break

        yield
