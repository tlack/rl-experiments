# CHANGE THIS:

opt_ranges = {
    "minibatch": [16],
    "learning_rate": [3e-6],
    "steps_per_episode": [1024],
    "n_update": [512],
    "entropy_coef": [0.01],
    "action_muting": [0.2],
    "n_episode": [500],
    "PPO_epochs": [16]
}

#

import datetime
import json
import random

import ppo_panda

def make_config(ranges): 
  return {k:random.choice(v) for k,v in ranges.items()}

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
