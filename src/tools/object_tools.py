import numpy as np

def get_distance(self, coordinates=None):
    length = np.sum((np.array(coordinates) - self.state["position"])**2, axis=-1)
    value = np.sqrt(np.min(length))
    idx = np.argmin(length)
    return value, idx

def move(self, coordinate=None):
    dx = coordinate[0] - self.state["position"][0]
    dy = coordinate[1] - self.state["position"][1]
    angle = np.arctan2(dy, dx)
    coordinate = np.array([np.cos(angle)*self.state["speed"], np.sin(angle)*self.state["speed"]])
    self.state["position"] = np.clip(self.state["position"] + coordinate, np.array([0, 0]), np.array([self.env_size, self.env_size]))
    return self.state