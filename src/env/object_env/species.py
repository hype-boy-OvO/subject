import numpy as np

class Species:
    def __init__(self, env_size=128, speed=1.0):
        self.speed = speed
        self.env_size = env_size
        self.state = self.reset()
        self.direction = [np.array([0, 1], dtype=float),np.array([1, 1], dtype=float),
                          np.array([1, 0], dtype=float),np.array([1, -1], dtype=float),
                          np.array([0, -1], dtype=float),np.array([-1, -1], dtype=float),
                          np.array([-1, 0], dtype=float),np.array([-1, 1], dtype=float)]
    
    def reset(self):
        return {"position": np.random.randint(low=0, high=self.env_size, size=(2,)).astype(float), "hunger": np.array([1.0], dtype=float), "speed": self.speed, "alive": True, "ate_feed": 0.0}

    def move(self, action):
        coordinate = self.direction[action] * 1.5 * self.state["speed"]
        self.state["position"] = np.clip(self.state["position"] + coordinate, np.array([0, 0]), np.array([self.env_size, self.env_size]))

    def step(self, action):
        if self.state["hunger"] <= 0:
            self.state["alive"] = False
        else:
            self.move(action)
