import numpy as np
from src.tools.object_tools import get_distance

class Feed:
    def __init__(self, env_size=None):
        self.env_size = env_size
        self.state = self.reset()
        self.get_distance = get_distance.__get__(self)

    def reset(self):
        state = {"position": np.random.randint(low=0, high=self.env_size, size=(2,)).astype(float)}
        self.lowest_distance = float('inf')
        return state

    def step(self, coordinates):
        value, idx = self.get_distance(coordinates)
        if value <= 2:
            return idx
        else:
            return None
