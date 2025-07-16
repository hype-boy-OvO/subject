import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict 
import numpy as np
from src.env.config import EnvConfig
from src.env.object_env import species, predator, feed
from src.tools import env_tools
import cv2

class CustomEnv(gym.Env):
    def __init__(self, config=None, render_mode=None, speciess=None, predators=None, feeds=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = Discrete(8)
        self.config = config if config is not None else EnvConfig()
        self.species_class = speciess if speciess is not None else species.Species
        self.predator_class = predators if predators is not None else predator.Predator
        self.feed_class = feeds if feeds is not None else feed.Feed
        self.is_margin = env_tools.is_margin
        self.make_position = env_tools.make_position
        self.observation_space = Dict({
            "feed_position": Box(low=0.0, high=1.0, shape=(5,5), dtype=float),
            "predator_position": Box(low=0.0, high=1.0, shape=(5,5), dtype=float),
            "is_margin": Box(low=0.0, high=1.0, shape=(2,), dtype=float)
            })

    def get_obs(self,mode=None):
        if mode == "species":
            return list(map(lambda s: s.state, self.species))

        if mode == "predator":
            return list(map(lambda p: p.state, self.predator))

        if mode == "feed":
            return list(map(lambda f: f.state, self.feed))
        
    def get_observation(self):
        species_state = self.get_obs("species")
        feed_state = self.get_obs("feed")
        predator_state = self.get_obs("predator")

        observation = [{
            "feed_position": np.array(self.make_position(i["position"], [f["position"] for f in feed_state])),
            "predator_position": np.array(self.make_position(i["position"], [p["position"] for p in predator_state])),
            "is_margin": np.array(self.is_margin(i["position"], self.config.env_size))
        } for i in species_state]

        if self.config.species.num == 1:
            observation = observation[0]

        return observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reward = 0.0
        self.species = [self.species_class(env_size=self.config.env_size, speed=self.config.species.speed) for _ in range(self.config.species.num)]
        self.feed = [self.feed_class(env_size=self.config.env_size) for _ in range(self.config.feed.num)]
        self.predator = [self.predator_class(env_size=self.config.env_size, speed=self.config.predator.speed) for _ in range(self.config.predator.num)]

        observation = self.get_observation()    
        info = {}
        return observation, info
    
    def info(self):
        return {}
    
    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False


        if len(self.species) == 1:
            self.species[0].step(action)
        else:
            for s, a in zip(self.species, action):
                s.step(a)

        species_state = self.get_obs("species")
        coordinates = [s["position"] for s in species_state]


        for f in self.feed:
            result = f.step(coordinates)
            if result is not None:
                reward += abs(1.0-self.species[result].state["hunger"]) + 4.0
                self.species[result].state["hunger"] = min(self.species[result].state["hunger"] + 0.3, 1.0)
                self.species[result].state["ate_feed"] += 1
                self.species[result].state["position"] = f.state["position"]
                f.reset()
            else:
                distance, idx = f.get_distance(coordinates)
                if distance < f.lowest_distance:
                    reward += distance**(-1)
                    f.lowest_distance = distance


        for p in self.predator:
            result = p.step(coordinates)
            if result is not None:
                self.species[result].state["alive"] = False


                

        observation = self.get_observation()
        length = len(self.species)
        self.species = [i for i in self.species if i.state["alive"]]
        reward -= (length-len(self.species))*15

        if len(self.species) == 0:
            terminated = True
        else:
            self.reward += 1
            reward += 0.1

        if self.reward >= 500.0:
            truncated = True
        

            
        
        info = self.info()

        if truncated or terminated:
            info['terminal_observation'] = observation

        return observation, reward, terminated, truncated, info
    

    def _render_frame(self):
        species_state = self.get_obs("species")
        feed_state = self.get_obs("feed")
        predator_state = self.get_obs("predator")
        screen = np.zeros((self.config.env_size, self.config.env_size, 3), dtype=np.uint8)
        for s in species_state:
            cv2.circle(screen, tuple(map(int, s["position"])), 1, (0, 0, 255), -1)

        for f in feed_state:
            cv2.circle(screen, tuple(map(int, f["position"])), 1, (0, 255, 0), -1)

        for p in predator_state:
            cv2.circle(screen, tuple(map(int, p["position"])), 1, (255, 0, 0), -1)

        return screen
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

