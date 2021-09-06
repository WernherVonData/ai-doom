import vizdoom as viz
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import experience_replay


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

    game.set_window_visible(False)

    actions = []
    for i in range(0, 7):
        actions.append([True if action_index == i else False for action_index in range(0, 7)])

    number_actions = len(actions)
    cnn = CNN(number_actions)
    softmax_body = SoftmaxBody(temperature=1.0)
    ai = AI(brain=cnn, body=softmax_body)

    n_steps = experience_replay.NStepProgress(game=game, ai=ai, n_step=10)
    memory = experience_replay.ReplayMemory(n_steps=n_steps, capacity=10000)

    ma = MA(100)

    # Training the AI
    loss = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)
    nb_epochs = 100
    for epoch in range(1, nb_epochs+1):
        memory.run_steps(200)
        for batch in memory.sample_batch(128):
            inputs, targets = eligibility_trace(cnn=cnn, batch=batch)
            inputs, tagets = Variable(inputs), Variable(targets)
            predictions = cnn(inputs)
            loss_error = loss(predictions, targets)
            optimizer.zero_grad()
            loss_error.backward()
            optimizer.step()
        rewards_steps = n_steps.rewards_steps()
        ma.add(rewards_steps)
        avg_reward = ma.average()
        print("Epoch {}, average reward: {}".format(epoch, avg_reward))

    # sleep_time = 1.0 / viz.DEFAULT_TICRATE  # = 0.028
    # episodes = 1
    # for i in range(episodes):
    #     game.new_episode()
    #     while not game.is_episode_finished():
    #         state = game.get_state()
    #         buffer = state.screen_buffer
    #         vars = state.game_variables
    #
    #         r = game.make_action(choice(actions))
    #
    #         print("State #" + str(i))
    #         print("Game variables:", vars)
    #         print("Reward:", r)
    #         print("=====================")
    #
    #         if sleep_time > 0:
    #             sleep(sleep_time)
    #
    #     # Check how the episode went.
    #     print("Episode finished.")
    #     print("Total reward:", game.get_total_reward())
    #     print("************************")
    # print("DONE")
