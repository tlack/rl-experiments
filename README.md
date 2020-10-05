# rl-experiments
Reinforcement learning experiments

## status

PPO + Panda sorta works on simple pushing task.

## contents

mypanda/: Hack of Panda environment with two diff goals - knocking the object around, or holding close to it

ppo_panda.py: PPO implementation targeting the Panda environment 

train.py: framework to iterate (potentially endlessly) over hyperparameters and
log results. this is from my "depressed and drifting" period

## how to use

Install pre-requisites:

```$ pip install -r requirements.txt```

Edit train.py to tweak the hyperparameters you want to try out (optional). Specify parameter options as lists, 
and training will try random combinations.

```$ vim train.py```

Set the system up to render live by creating `/tmp/render.txt`. Remove this file to turn off the GUI's rendering, 
which is faster.

```$ touch /tmp/render.txt```

Start training! The system will collect statistics in `./train-log.jsonlines`.

```$ python train.py```

Weights will be saved every 1000 episodes to `panda_saved`.

## origins

PyBullet Panda simulation: (unknown)

PPO implementation: https://github.com/wisnunugroho21/reinforcement_learning_ppo_rnd

