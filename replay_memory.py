import numpy as np
from collections import deque


class ReplayMemory:
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
        while (ofs + 1) * batch_size <= len(self.buffer):
            yield vals[ofs * batch_size:(ofs + 1) * batch_size]
            ofs += 1

    def append_memory(self, data):
        self.buffer.append(data)
        while len(self.buffer) > self.capacity:
            self.buffer.popleft()

    def is_buffer_full(self):
        # return True
        return len(self.buffer) >= self.capacity