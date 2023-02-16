import gym
from sample_factory.envs.env_wrappers import PixelFormatChwWrapper
#import minihack

# noinspection PyUnusedLocal
def make_minihack_env(env_name, cfg=None, **kwargs):
    env = gym.make(env_name, observation_keys=["pixel"])
    env = PixelFormatChwWrapper(env)
    #env = gym.make(env_name)
    """ env = gym.make(
    "MiniHack-River-v0",
    observation_keys=["pixel"]
    )  """
    return env
