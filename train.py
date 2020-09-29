# CHANGE THIS:

opt_ranges = {
    "n_episode": [50],
    "n_update": [32, 99, 299, 499, 999, 1499],
    "learning_rate": [3e-1, 3e-2, 3e-3, 3e-4, 3e-5, 3e-6],
    "goal_type": ["bumps"]
}

#

import datetime
import json
import random

import ppo_panda

def make_config(ranges): return {k:random.choice(v) for k,v in ranges.items()}
while 1:
    cfg = make_config(opt_ranges)
    print(f"!!** TRAINING NEW CONFIG:\n\n{repr(cfg)}\n\n")
    reward = ppo_panda.main(**cfg)
    reward["date"] = datetime.datetime.today()
    reward["config"] = cfg
    j = {k:repr(v) for k,v in reward.items()}
    open("train-log.jsonlines", "a").write(json.dumps(j)+"\n")


