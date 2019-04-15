import GridWorldEnv as GWEnv
import json
import numpy as np
env = GWEnv.GridWorldEnv('config.json', 1)
with open('config.json') as f:
    config = json.load(f)    

env.reset()

for i in range(100):
    actions = [1 for _ in range(config['N'])]
    obs, rewards, done, info = env.step(np.array(actions))
    print(obs)
    print(rewards)
    print(done)
    print(info)
