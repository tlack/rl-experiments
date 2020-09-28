from gym.envs.registration import register

register(
    id='panda-v9',
    entry_point='mypanda.envs:PandaEnv',
)
