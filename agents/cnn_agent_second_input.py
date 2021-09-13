import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import agents.cnn_agent


class CnnTwoInput(agents.cnn_agent.CNN):

    def __init__(self, number_actions, image_dim, linear_input):
        super(CnnTwoInput, self).__init__(number_actions, image_dim)
        self.fcLinear = nn.Linear(in_features=linear_input, out_features=20)
        self.fc2 = nn.Linear(in_features=60, out_features=number_actions)

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
        hp = F.relu(self.fcLinear(input_linear))
        combined = torch.cat((x, hp), dim=1)
        return self.fc2(combined)
