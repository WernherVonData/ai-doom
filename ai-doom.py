import vizdoom as viz
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from time import sleep

# Building the AI
import image_preprocessing

#TODO: Own experience replay


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
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.fc1 = nn.Linear(in_features=self.count_neurons((1, 80, 80)), out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=number_actions)

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
        probabilities = F.softmax(outputs * self.temperature)
        actions = probabilities.multinomial(0)
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


def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32))


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

    actions = []
    for i in range(0, 7):
        actions.append([True if action_index == i else False for action_index in range(0, 7)])

    number_actions = len(actions)

    episodes = 1
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            buffer = state.screen_buffer
            image_preprocessing.process_image_to_grayscale(buffer, 80, 80)
    print("DONE")
# # Getting the Doom environment
# doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
# doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
# number_actions = doom_env.action_space.n
#
# # Building an AI
# cnn = CNN(number_actions)
# softmax_body = SoftmaxBody(T = 1.0)
# ai = AI(brain = cnn, body = softmax_body)
#
# # Setting up Experience Replay
# n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10)
# memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)
