
from sample_factory.algorithms.appo.appo import APPO

env_id = "gym_CartPole-v0"
device ="gpu"

model = APPO(env=env_id, device=device, 
             num_workers=2,
             num_envs_per_worker=8,
             encoder='mlp', encodersubtype = 'mlp_mujoco')
    
model.train(train_for_env_steps=1000000)