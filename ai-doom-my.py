import vizdoom as viz
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os
from collections import deque, namedtuple
import image_preprocessing
import matplotlib.pyplot as plt

Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])


def get_gpu():
    if torch.cuda.is_available():
        print("***** CUDA IS HERE")
        return 1
    print("*****NO CUDA")
    return -1


class CNN(nn.Module):

    def __init__(self, number_actions):
        super(CNN, self).__init__()
        get_gpu()
        # We will work with black and white images
        # Out channels - number of features we want to detect
        # mean the number of images with applied convolution
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)
        self.fc1 = nn.Linear(in_features=self.count_neurons((1, 80, 80)), out_features=80)
        self.fc2 = nn.Linear(in_features=80, out_features=number_actions)

    def count_neurons(self, image_dim):
        # 1- batch, image_dim - dimensions of the image - channels, width, height
        # * - to pass tuple argument as a list of arguments to a function
        x = Variable(torch.rand(1, *image_dim))
        # 3 - kernel size
        # 2 - stride - how many pixels it's going to slide into the images
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        # Get neurons from all the channels and put them into the one vector
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SoftmaxBody(nn.Module):

    def __init__(self, temperature):
        super(SoftmaxBody, self).__init__()
        self.temperature = temperature

    # Outputs from the neural network
    def forward(self, outputs):
        probabilities = F.softmax(outputs * self.temperature, dim=len(outputs))
        actions = probabilities.multinomial(1)
        return actions


class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input_images = Variable(torch.from_numpy(np.array(inputs, dtype=np.float32)))
        output = self.brain(input_images)
        actions = self.body(output)
        return actions.data.numpy()


def eligibility_trace(cnn, batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32))
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)


class MA:
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


class Memory:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque()

    def sample_batch(self, batch_size):
        """
        Creates an iterator that returns random batches
        :param batch_size: batch size
        :return: iterator returning random batches
        """
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs+1)*batch_size <= len(self.buffer):
            yield vals[ofs*batch_size:(ofs+1)*batch_size]
            ofs += 1

    def append_memory(self, data):
        self.buffer.append(data)
        while len(self.buffer) > self.capacity:
            self.buffer.popleft()


def save(filename, model, optimizer):
    torch.save({'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, filename)


def load(self):
    if os.path.isfile('last_brain.pth'):
        print("=>loading checkpoint")
        checkpoint = torch.load('last_brain.pth')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("no checkpoint found...")


if __name__ == '__main__':
    game = viz.DoomGame()
    game.load_config("scenarios/deadly_corridor.cfg")
    game.init()

    game.add_available_button(viz.Button.MOVE_LEFT)
    game.add_available_button(viz.Button.MOVE_RIGHT)
    game.add_available_button(viz.Button.ATTACK)
    game.add_available_button(viz.Button.MOVE_FORWARD)
    game.add_available_button(viz.Button.MOVE_BACKWARD)
    game.add_available_button(viz.Button.TURN_LEFT)
    game.add_available_button(viz.Button.TURN_RIGHT)

    game.set_window_visible(True)
    game.set_episode_timeout(0)

    actions = []
    for i in range(0, 7):
        actions.append([True if action_index == i else False for action_index in range(0, 7)])

    number_actions = len(actions)
    cnn = CNN(number_actions)
    softmax_body = SoftmaxBody(temperature=1.0)
    ai = AI(brain=cnn, body=softmax_body)

    ma = MA(100)

    # Training the AI
    loss = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)
    nb_epochs = 100
    nb_steps = 200
    training_not_finished = True
    current_step = 0
    rewards = []
    epoch = 1
    memory = Memory(capacity=10000)
    reward = 0.0
    history_reward = []
    history = deque()
    replay_memory_tuple_size = 15
    while training_not_finished:
        if game.is_episode_finished():
            game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            buffer = state.screen_buffer
            img = image_preprocessing.process_image_to_grayscale(buffer, 80, 80)
            action_idx = ai(np.array([img]))[0][0]
            r = game.make_action(actions[action_idx])
            reward += r
            history.append(Step(state=img, action=action_idx, reward=r, done=game.is_episode_finished()))
            if len(history) > replay_memory_tuple_size:
                memory.append_memory(tuple(history))
                history = deque()
            current_step += 1
            rewards.append(r)
            history_reward.append(r)
            if len(history_reward) >= 10000:
                history_reward = history_reward[1:]
            if current_step == nb_steps:
                current_step = 0
                # print("Current buffer size: {}".format(len(memory.buffer)))
                optimization_happened = False
                for batch in memory.sample_batch(128):
                    optimization_happened = True
                    inputs, targets = eligibility_trace(cnn=cnn, batch=batch)
                    inputs, targets = Variable(inputs), Variable(targets)
                    predictions = cnn(inputs)
                    loss_error = loss(predictions, targets)
                    optimizer.zero_grad()
                    loss_error.backward()
                    optimizer.step()
                if optimization_happened:
                    rewards_steps = rewards
                    rewards = []
                    ma.add(rewards_steps)
                    avg_reward = ma.average()
                    print("Epoch {}, average reward: {}".format(epoch, avg_reward))
                    if epoch % 10 == 0:
                        model_file = "results\cnn_doom_"+str(epoch)+".pth"
                        score_file = "results\scores_" + str(epoch) + ".png"
                        print("Saving model file: {} and diagram: {}".format(model_file, score_file))
                        plt.plot(history_reward, color='blue')
                        plt.savefig(score_file)
                        save(model_file, cnn, optimizer)
                    epoch += 1

            if epoch >= nb_epochs:
                training_not_finished = False
                game.close()
                break
