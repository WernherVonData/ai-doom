import vizdoom as vzd
import torch
from torch.autograd import Variable
from agents import cnn_agent
import image_preprocessing
import numpy as np
from time import sleep
import utils
from softmax_body import SoftmaxBody


class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input_images = Variable(torch.from_numpy(np.array(inputs, dtype=np.float32))).to(utils.DEVICE_NAME)
        output = self.brain(input_images)
        actions = self.body(output)
        return actions.data.cpu().numpy()


if __name__ == '__main__':
    scenario = "scenarios/basic.cfg"
    print("=>device used: {}".format(utils.DEVICE_NAME))

    actions = []
    nb_available_buttons = 3
    for i in range(0, nb_available_buttons):
        actions.append([True if action_index == i else False for action_index in range(0, nb_available_buttons)])
    number_actions = len(actions)
    image_dim = 128

    cnn = cnn_agent.CNN(number_actions=nb_available_buttons, image_dim=image_dim)
    cnn = utils.load("experiments\\basic_scenario\\basic_cnn_doom_50.pth", cnn)
    cnn.to(utils.DEVICE_NAME)
    softmax_body = SoftmaxBody(temperature=1.0)
    ai = AI(brain=cnn, body=softmax_body)
    game = vzd.DoomGame()
    game.load_config(scenario)
    game.init()
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028
    nb_episodes = 50
    for episode in range(1, nb_episodes+1):
        game.new_episode()
        reward = 0
        while not game.is_episode_finished():
            state = game.get_state()
            buffer = state.screen_buffer
            img = image_preprocessing.process_image_to_grayscale(buffer, image_dim, image_dim)
            action = ai(np.array([img]))[0][0]
            reward += game.make_action(actions[action])
            if sleep_time > 0:
                sleep(sleep_time)
        print("Episode {}, reward: {}".format(episode, reward))
