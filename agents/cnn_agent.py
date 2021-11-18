import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import utils

class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input_images = Variable(torch.from_numpy(np.array(inputs, dtype=np.float32))).to(utils.DEVICE_NAME)
        output = self.brain(input_images)
        actions = self.body(output)
        return actions.data.cpu().numpy()


class CNN(nn.Module):

    def __init__(self, number_actions, image_dim, hidden_conv_channels=64, out_conv_channels=128, hidden_linear_size=80):
        super(CNN, self).__init__()
        # We will work with black and white images
        # Out channels - number of features we want to detect
        # mean the number of images with applied convolution
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=hidden_conv_channels, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=hidden_conv_channels, out_channels=hidden_conv_channels, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels=hidden_conv_channels, out_channels=out_conv_channels, kernel_size=2)
        self.fc1 = nn.Linear(in_features=self.count_neurons((1, image_dim, image_dim)), out_features=hidden_linear_size)
        self.fc2 = nn.Linear(in_features=hidden_linear_size, out_features=number_actions)

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

    def forward(self, input_img):
        x = F.relu(F.max_pool2d(self.convolution1(input_img), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
