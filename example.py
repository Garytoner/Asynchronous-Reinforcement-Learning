from Asynchronous_Reinforcement_Learning.algorithms.appo.appo import APPO
def main():
    env_id = "gym_CartPole-v0"
    device ="gpu"
    model = APPO(env=env_id, device=device, 
                 num_workers=2,
                 num_envs_per_worker=8,
                 encoder='mlp', encodersubtype = 'mlp_mujoco')
        
    model.train(train_for_env_steps=100000)
 
if __name__ == '__main__':
    main()