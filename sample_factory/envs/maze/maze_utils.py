import gym

# noinspection PyUnusedLocal
def make_maze_env(env_name, cfg=None, **kwargs):
    env = gym.make(env_name)
    return env

