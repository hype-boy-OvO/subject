from src.env.config import EnvConfig
from src.simulate.simul_object.species import Species
from src.env.env import CustomEnv
from sb3_contrib import RecurrentPPO
import numpy as np
import json
import random

config = EnvConfig()
config.species.num=18
config.predator.num=3
config.feed.num=5

env = CustomEnv(config=config, speciess=Species)
model = RecurrentPPO.load("src/ppo/logs/best_model/best_model.zip", env=env)

state = None

episode_start = np.array([True])
obs, info = env.reset()

for o, s in zip(obs, env.species):
    s.obs = o
    s.statel = None
    s.episode_start = np.array([True])


for i in range(30):

    data= list(map(lambda s: {"life":s["life"], "speed":s["speed"]}, env.get_obs(mode="species")))
    with open(f"datas/data{i}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    while len(env.species) >= 13:
        for s in env.species:
            s.action, s.statel = model.predict(s.obs, state=s.statel, episode_start=s.episode_start, deterministic=True)
            s.step(s.action)
        species_state = env.get_obs("species")
        coordinates = [s["position"] for s in species_state]
        reward = 0


        for f in env.feed:
            result = f.step(coordinates)
            if result is not None:
                reward += abs(1.0-env.species[result].state["hunger"]) + 4.0
                env.species[result].state["hunger"] = min(env.species[result].state["hunger"] + 0.3, 1.0)
                env.species[result].state["ate_feed"] += 1
                env.species[result].state["position"] = f.state["position"]
                f.reset()

        for p in env.predator:
            result = p.step(coordinates)
            if result is not None:
                env.species[result].state["alive"] = False

        terminated = False
        truncated = False


        obs = env.get_observation()
        length = len(env.species)
        env.species = [i for i in env.species if i.state["alive"]]

    random.shuffle(env.species)
    sp_num = len(env.species)

    while True:
        for i in range(0, sp_num, 2):
            species = Species(parents_genes=[env.species[i].genes,env.species[i+1].genes])
            env.species.append(species)
            if len(env.species) >= 12:
                break
        if len(env.species) >= 12:
            break
    
    obs = env.get_observation()

    for o, s in zip(obs, env.species):
        s.obs = o
        s.statel = None
        s.episode_start = np.array([True])