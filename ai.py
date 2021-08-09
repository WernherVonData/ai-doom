import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import gym

# Building the AI

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
        self.fc1 = nn.Linear(in_features=number_neurons, out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=number_actions)


# Deep Q-Learning implementation