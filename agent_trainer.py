from collections import deque
import datetime
from random import choice

import matplotlib.pyplot as plt

import vizdoom as viz

import utils

from katie.rl import ReplayMemory
from utils import MemoryAverage

# Provides definitions of the objects that are serialized by this script
from agents import history_records


class AgentTrainer:
    def __init__(self, agent_to_train, memory_capacity=100000):
        self.agent = agent_to_train
        self.ma = MemoryAverage(100)
        self.memory = ReplayMemory(capacity=memory_capacity)

    def train(self, starting_epoch=1, nb_epochs=100, batch_size=128, n_step=10, n_steps=1000, gamma=0.99, memory_path=None):
        """
        Train the agent given to the trainer. The trainer will first fill the memory up to it's capacity using the
        received agent. After that the agent training will continue.

        Every 10 epochs the agent and training memory are saved. Also the scoring plots are generated.
        :param starting_epoch: number of the epoch that we are going to start.
        :param nb_epochs: for how many epochs we are going to train the agent.
        :param batch_size: number of samples taken in each training step.
        :param n_step: how many steps the agent will perform in the environment before saving them as a memory record.
        :param n_steps: how many records will be saved to the memory before training will happen.
        :param gamma: gamma parameter of the agent eligibility trace algorithm.
        :param memory_path: path to the memory file, is set to None the learning will start with empty memory
        :return: None
        """
        game = viz.DoomGame()
        game.load_config(self.agent.scenario_path)
        game.init()
        game.new_episode()

        rewards = []
        history_reward = []
        avg_history_reward = []
        epoch = starting_epoch

        if memory_path is not None:
            self.memory.load_memory_buffer(memory_path)

        while True:
            if game.is_episode_finished():
                game.new_episode()
            histories = []
            history = deque()
            reward = 0.0
            while True:
                state_data = self.agent.read_game_data(game=game)
                action = self.agent.make_action(state_data=state_data) if self.memory.is_buffer_full() else choice(
                    range(0, self.agent.nb_available_buttons))
                game_reward = game.make_action(self.agent.actions[action])
                reward += self.agent.calculate_reward(game_reward=game_reward)
                history.append(self.agent.generate_history_record(action=action, game_finished=game.is_episode_finished()))
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
                if len(histories) >= n_steps:
                    break
            for history in histories:
                self.memory.append_memory(history)
            if not self.memory.is_buffer_full():
                print("Memory not full {} of {}".format(self.memory.get_current_buffer_size(),
                                                        self.memory.get_capacity()))
                continue
            start = datetime.datetime.now()
            for batch in self.memory.sample_batch(batch_size):
                self.agent.perform_training_step(batch=batch, gamma=gamma)
            stop = datetime.datetime.now()
            delta = stop - start
            rewards_steps = rewards
            history_reward = history_reward + rewards
            rewards = []
            self.ma.add(rewards_steps)
            avg_reward = self.ma.average()
            avg_history_reward.append(avg_reward)
            print("Epoch: %s, Average Reward: %s Delta: %f Time: %s" % (
                str(epoch), str(avg_reward), delta.seconds, str(datetime.datetime.now())))
            epoch += 1
            if epoch % 10 == 0:
                model_file = "results\\cnn_doom_" + self.agent.agent_identifier + "_" + str(epoch) + ".pth"
                score_file = "results\\scores_" + self.agent.agent_identifier + "_" + str(epoch) + ".png"
                avg_score_file = "results\\avg_scores_" + self.agent.agent_identifier + "_" + str(epoch) + ".png"
                memory_file = "results\\buffer_" + self.agent.agent_identifier + "_" + str(epoch) + ".pickle"
                self.memory.save_memory_buffer(memory_file) # Currently giving an error
                print("Saving model file: {} and diagram: {}".format(model_file, score_file))
                plt.clf()
                plt.plot(history_reward, color='blue')
                plt.savefig(score_file)
                plt.clf()
                plt.plot(avg_history_reward, color='green')
                plt.savefig(avg_score_file)
                utils.save(model_file, self.agent.cnn, self.agent.optimizer)
            if epoch == nb_epochs:
                print("Reached last epoch, finishing...")
                break
