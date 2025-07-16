import numpy as np
import random
from src.tools import object_tools

class Predator:
    def __init__(self, env_size=None, speed=1.0):
        position = np.array([random.choice([0.0, float(env_size)]), float(random.randint(0, env_size))], dtype=float)
        np.random.shuffle(position)
        self.state = {"position": position,"speed": speed}
        self.env_size = env_size
        self.get_distance = object_tools.get_distance.__get__(self)
        self.move = object_tools.move.__get__(self)


    def step(self, coordinates):
        value, idx = self.get_distance(coordinates)
        coordinate = coordinates[idx]
        if value <= self.state["speed"]:
            self.state["position"] = coordinate
            return idx
        else:
            self.move(coordinate)
            return None
