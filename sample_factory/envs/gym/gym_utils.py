import gym

class GymSpec:
    def __init__(self, name, env_id):
        self.name = name
        self.env_id = env_id


GYM_ENVS = [
    GymSpec('gym_CartPolev0', 'CartPole-v0'),
    GymSpec('gym_CartPolev1', 'CartPole-v1'),
]


def gym_env_by_name(name):
    for cfg in GYM_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown Gym env')


def make_gym_env(env_name, cfg, **kwargs):
    mujoco_spec = gym_env_by_name(env_name)
    env = gym.make(mujoco_spec.env_id)
    return env
