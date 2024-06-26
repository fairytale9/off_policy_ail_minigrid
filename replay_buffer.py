import collections
import numpy as np

Experience = collections.namedtuple('Experience', field_names=['state', 'action'])
Transition = collections.namedtuple('Transition', field_names=['state', 'action', 'next_state'])

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions)

class TransitionReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions)
