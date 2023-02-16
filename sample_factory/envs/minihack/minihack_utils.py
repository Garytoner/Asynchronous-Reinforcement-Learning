import gym
from sample_factory.envs.env_wrappers import PixelFormatChwWrapper
from sample_factory.envs.minihack.minihack_model import minihack_register_models

def model_initialized(cfg, env_name):
    minihack_register_models()

# noinspection PyUnusedLocal
def make_minihack_env(env_name, cfg=None, **kwargs):
    model_initialized(cfg, env_name)
    env = gym.make(env_name, observation_keys=["pixel"])
    env = PixelFormatChwWrapper(env)
    return env
