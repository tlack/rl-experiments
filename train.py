# CHANGE THIS:

opt_ranges = {
    "learning_rate": [3e-5, 3e-7, 3e-9],
    "steps_per_episode": 256,
    "action_muting": [0.2, 0.1, 0.05],
    "n_episode": 5 * 1000,
    "n_update": 256,
    "minibatch": 64,
    "fixed_seed": 777,
    "PPO_epochs": 8
}

#

import datetime
import json
import random

import panda_env_v9
import ppo_panda

picker_seed = 97779

def make_config(ranges): 
  global picker_seed
  picker_seed += 1
  random.seed(picker_seed)
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

# Baseline note:
# 
# With fixed_seed = 777, and configuration:
#
#   {'learning_rate': 3e-07, 'steps_per_episode': 200, 'action_muting': 0.25, 'n_episode': 20, 'n_update': 256, 'minibatch': 64, 'fixed_seed': 777}
#
# ..should bump every time in about 74 steps:
#
# BUMPED! 1.0653640226516654
# Episode 1 	 t_reward: -13.969270709717811 	 steps: 37
# BUMPED! 1.054997803839964
# Episode 2 	 t_reward: -37.73644336278778 	 steps: 74
#

