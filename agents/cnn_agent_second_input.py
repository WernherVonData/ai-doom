import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import agents.cnn_agent
import utils


class AITwoInput:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, image_inputs, linear_inputs):
        input_images = Variable(torch.from_numpy(np.array(image_inputs, dtype=np.float32))).to(utils.DEVICE_NAME)
        linear_input = Variable(torch.from_numpy(np.array(linear_inputs, dtype=np.float32))).to(utils.DEVICE_NAME)
        output = self.brain(input_images, linear_input)
        actions = self.body(output)
        return actions.data.cpu().numpy()


class CnnTwoInput(agents.cnn_agent.CNN):

    def __init__(self, number_actions, image_dim, linear_input, hidden_conv_channels=64, out_conv_channels=128, hidden_linear_size=80, second_layer_size=30):
        super(CnnTwoInput, self).__init__(number_actions, image_dim, hidden_conv_channels, out_conv_channels)
        self.fcLinear = nn.Linear(in_features=linear_input, out_features=second_layer_size)
        self.fc2 = nn.Linear(in_features=hidden_linear_size+second_layer_size,  out_features=number_actions)

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

    def forward(self, input_img, input_linear):
        x = F.relu(F.max_pool2d(self.convolution1(input_img), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        linear = input_linear.reshape(input_linear.size(0), -1)
        linear = F.relu(self.fcLinear(linear))
        combined = torch.cat((x, linear), dim=1)
        return self.fc2(combined)
