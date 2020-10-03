import gym
import panda_env_v10 as panda

def build_ui(env):
    dv = 1
    r = [env._p.addUserDebugParameter(q, -dv, dv, 0) for q in ["x", "y", "z", "f"]]
    return r

def main():
    env = gym.make('panda-v10')
    env.steps_per_episode = 9e9
    env.reset()
    sliders = build_ui(env)
    while 1:
        action = [env._p.readUserDebugParameter(q) for q in sliders]
        print('action',action)
        state, reward, done, info = env.step(action)
        print('state', state)
        print('reward', reward)
        if done == True: print('DONE!')
        if info: print('info', info)
        if hasattr(env, 'getExtendedObservation'):
            obs = env.getExtendedObservation()
            if obs: print('ext_obs', obs)

main()
