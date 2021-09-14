import vizdoom as viz
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from collections import deque, namedtuple
import image_preprocessing
import matplotlib.pyplot as plt
import datetime
from agents import cnn_agent
import utils
from utils import MemoryAverage
from softmax_body import SoftmaxBody
from replay_memory import ReplayMemory

Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])

print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")


class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input_images = Variable(torch.from_numpy(np.array(inputs, dtype=np.float32))).to(utils.DEVICE_NAME)
        output = self.brain(input_images)
        actions = self.body(output)
        return actions.data.cpu().numpy()


def eligibility_trace(cnn, batch, gamma=0.99):
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


if __name__ == '__main__':
    game = viz.DoomGame()
    game.load_config("scenarios/basic.cfg")
    game.init()

    game.set_window_visible(True)

    lr = 0.001
    nb_epochs = 100
    nb_steps = 200
    image_dim = 128
    gamma = 0.99
    memory_capacity = 10000
    n_step = 10
    batch_size = 64
    temperature = 1.0

    actions = []
    nb_available_buttons = 3
    for i in range(0, nb_available_buttons):
        actions.append([True if action_index == i else False for action_index in range(0, nb_available_buttons)])
    number_actions = len(actions)

    cnn = cnn_agent.CNN(number_actions=number_actions, image_dim=image_dim)
    cnn.to(utils.DEVICE_NAME)
    softmax_body = SoftmaxBody(temperature=temperature)
    ai = AI(brain=cnn, body=softmax_body)

    ma = MemoryAverage(100)

    # Training the AI
    loss = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    memory = ReplayMemory(capacity=memory_capacity)

    rewards = []
    reward = 0.0
    history_reward = []
    avg_history_reward = []
    history = deque()

    game.new_episode()
    epoch = 1
    while True:
        if game.is_episode_finished():
            game.new_episode()
        histories = []
        history = deque()
        reward = 0.0
        while True:
            state = game.get_state()
            buffer = state.screen_buffer
            img = image_preprocessing.to_grayscale_and_resize(buffer, image_dim, image_dim)
            action = ai(np.array([img]))[0][0]
            r = game.make_action(actions[action])
            reward += r
            history.append(Step(state=img, action=action, reward=r, done=game.is_episode_finished()))
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
            if len(histories) >= nb_steps:
                break
        for history in histories:
            memory.append_memory(history)
        start = datetime.datetime.now()
        for batch in memory.sample_batch(batch_size):
            inputs, targets = eligibility_trace(cnn=cnn, batch=batch, gamma=gamma)
            inputs, targets = Variable(inputs).to(utils.DEVICE_NAME), Variable(targets)
            predictions = cnn(inputs)
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
        print("Epoch: %s, Average Reward: %s Delta: %f" % (str(epoch), str(avg_reward), delta.seconds))
        epoch += 1
        if epoch % 10 == 0:
            model_file = "results\\basic_cnn_doom_" + str(epoch) + ".pth"
            score_file = "results\\basic_scores_" + str(epoch) + ".png"
            avg_score_file = "results\\basic_avg_scores_" + str(epoch) + ".png"
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
