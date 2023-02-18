import sys

from Asynchronous_Reinforcement_Learning.algorithms.utils.arguments import default_cfg
from Asynchronous_Reinforcement_Learning.envs.create_env import create_env
from Asynchronous_Reinforcement_Learning.utils.utils import log


def main():
    env_name = 'doom_battle'
    env = create_env(env_name, cfg=default_cfg(env=env_name))

    env.reset()
    done = False
    while not done:
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())

    log.info('Done!')


if __name__ == '__main__':
    sys.exit(main())
