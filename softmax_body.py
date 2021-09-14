import torch.nn as nn
import torch.nn.functional as F


class SoftmaxBody(nn.Module):

    def __init__(self, temperature):
        super(SoftmaxBody, self).__init__()
        self.temperature = temperature

    def forward(self, outputs):
        probabilities = F.softmax(outputs * self.temperature, dim=len(outputs))
        actions = probabilities.multinomial(num_samples=1)
        return actions
