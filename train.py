# CHANGE THIS:

opt_ranges = {
    "learning_rate": [3e-7],
    "action_muting": [0.2],
    "n_episode": [30],
    "steps_per_episode": [500]
}

#

import datetime
import json
import random

import panda_env_v9
import ppo_panda

def make_config(ranges): 
  return {k:(random.choice(v) if type(v) == type([]) else v) for k,v in ranges.items()} 

def train_with_config(cfg):
  reward = ppo_panda.main(**cfg)
  reward["date"] = datetime.datetime.today()
  reward["config"] = cfg
  j = {k:repr(v) for k,v in reward.items()}
  open("train-log.jsonlines", "a").write(json.dumps(j)+"\n")
  return reward

def train_with_ranges(opt_ranges):
  cfg = make_config(opt_ranges)
  print(f"!!** TRAINING NEW CONFIG:\n\n{repr(cfg)}\n\n")
  train_with_config(cfg)

if __name__ == "__main__":
  while 1:
    train_with_ranges(opt_ranges)
