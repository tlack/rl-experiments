# CHANGE THIS:

opt_ranges = {
    "n_updates": [50, 99, 299, 499, 999, 1499],
    "goal_types": ["touches"]
}

#

import random
import ppo_panda

def make_config(ranges): return {k:random.choice(v) for k,v in ranges.items()}
while 1:
    cfg = make_config(opt_ranges)
    print(f"!!** TRAINING NEW CONFIG:\n\n{repr(cfg)}\n\n")
    ppo_panda.main(**cfg)

