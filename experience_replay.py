import vizdoom as viz
import numpy as np
from collections import deque, namedtuple
import image_preprocessing
from time import sleep

Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])


class NStepProgress:

    def __init__(self, game, ai, n_step):
        self.ai = ai
        self.rewards = []
        self.game = game
        self.n_step = n_step

    def __iter__(self):
        self.game.new_episode()
        history = deque()
        reward = 0.0
        sleep_time = 1.0 / viz.DEFAULT_TICRATE  # = 0.028
        while True:
            state = self.game.get_state()
            buffer = state.screen_buffer
            img = image_preprocessing.process_image_to_grayscale(buffer, 80, 80)
            action = self.ai(np.array([img]))[0][0]
            print("Action: ", action)
            r = self.game.make_action([action])
            reward += r
            history.append(Step(state=img, action=action, reward=r, done=self.game.is_episode_finished()))
            print("State #" + str(state.number))
            print("Game variables:", state.game_variables)
            print("Reward:", r)
            print("=====================")
            # if sleep_time > 0:
            #     sleep(sleep_time)
            while len(history) > self.n_step+1:
                yield tuple(history)
            if self.game.is_episode_finished():
                if len(history) > self.n_step+1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                self.rewards.append(reward)
                reward = 0.0
                self.game.new_episode()
                history.clear()

    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps


class ReplayMemory:

    def __init__(self, n_steps, capacity=10000):
        self.capacity = capacity
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
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

    def run_steps(self, samples):
        while samples > 0:
            entry = next(self.n_steps_iter)
            self.buffer.append(entry)
            samples -= 1
        while len(self.buffer) > self.capacity:
            self.buffer.popleft()
