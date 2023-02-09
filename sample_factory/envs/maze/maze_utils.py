import gym

# noinspection PyUnusedLocal
def make_maze_env(env_name, cfg=None, **kwargs):
    env = gym.make(env_name)
    env = gym.wrapper.RecordEpisodeStatistics(env)
    return env

