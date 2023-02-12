import gym
import gym_maze
from sample_factory.envs.maze.maze_model import maze_register_models

def model_initialized(cfg, env_name):
    maze_register_models()


def make_maze_env(env_name, cfg=None, **kwargs):
    model_initialized(cfg, env_name)
    env = gym.make(env_name)
    return env

