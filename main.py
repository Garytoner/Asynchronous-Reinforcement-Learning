import argparse
import os
from distutils.util import strtobool
from sample_factory.algorithms.appo.appo import APPO,A3C,IMPALA
import torch
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1223,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with tensorboard")
    # parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
    #     help="the wandb's project name")
    # parser.add_argument("--wandb-entity", type=str, default=None,
    #     help="the entity (team) of wandb's project")
    # parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #     help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="mujoco_CartPolev0",
        help="the id of the environment")
    parser.add_argument("--num_workers", type=int, default=32,
        help="the id of the environment")
    parser.add_argument(
            '--num_envs_per_worker', default=32, type=int,
            help='Number of envs on a single CPU actor, in high-throughput configurations this should be in 10-30 range for Atari/VizDoom '
                    'Must be even for double-buffered sampling!')
    parser.add_argument("--no-obs-norm",type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="normali")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument('--encoder_type', default='mlp', type=str, help='Type of the encoder. Supported: conv, mlp, resnet (feel free to define more)')
    parser.add_argument('--encoder_subtype', default='mlp_mujoco', type=str, help='Specific encoder design (see model.py)')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = "gpu"
    args.env_id = "atari_pong"
    args.encoder_type ="conv"
    args.encoder_subtype ="convnet_test"
   # args.env_id = "mujoco_nasimlarge"
   # device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    #envs = make_vec_envs(args.env_id, args.seed, args.num_envs, args.gamma)
    #envs = make_vec_envs_sb(args.env_id, n_envs=args.num_envs, seed=45821)
   # params = {0: "./train_dir/default_experiment/checkpoint_p0",1: "./train_dir/default_experiment/checkpoint_p1",2: "./train_dir/default_experiment/checkpoint_p2"}
    #model.set_parameters(params)
    model = APPO(env=args.env_id, device=device, num_workers=args.num_workers,num_envs_per_worker=args.num_envs_per_worker,encoder=args.encoder_type,encodersubtype=args.encoder_subtype,policy_kwargs = {"num_policies":1,"with_pbt":False})
   # model.train()
   # params = model.get_parameters()
   # print(params)  
    params = {0: "./train_dir/default_experiment/checkpoint_p0"}  
   # model.train(200000)
   # model.set_parameters(params)
   # params = model.get_parameters()
   # print(params)
   #  PPO_SB("MlpPolicy", env=envs, create_eval_env=True, verbose=0)
    # model.learn(total_timesteps=400000)
    # mean_reward, std_reward = evaluate_policy(model, gym.make(args.env_id), n_eval_episodes=10)
    # print(mean_reward)
   # params = {0: "./train_dir/default_experiment/checkpoint_p0"}
   # model.set_parameters(params)
    #e_model = APPO(envs=args.env_id, device=device, num_envs=args.num_envs, verbose=1)#,  num_steps=256, update_epochs=4, num_minibatches=4, ent_coef=0.02)
    #print("start training")
   # model.train(200000000)
    #mean_reward, std_reward=e_model.eval(num_eval_episodes=10)
    #print(mean_reward)
    #model.eval(num_eval_episodes=2)
    #params = model.get_parameters()
   # print(params)
   # params = {0: "./train_dir/default_experiment/checkpoint_p0"}
    #model.set_parameters(params)
    #print("reset")
   # params = {0: "./train_dir/default_experiment/checkpoint_p0"}
   # params = model.get_parameters()
   # print(params)
   # model.train(20000000)
    #params = model.get_parameters()
   # print(params)
   # model.set_parameters(params)
   # params = model.get_parameters()
   # print(params)
   # model.train(20000000)
   # #model.eval(num_eval_episodes=10)
   # model.train(10000000000)
   # model.train(100000)
   
   
    model.train(10000000)
    model.train(10000000)
    model.train(10000000)
    model.train(10000000)
    model.train(10000000)
    params = model.get_parameters()
    print(params)
    model.train(10000000)
    model.train(10000000)
    model.train(10000000)
    model.train(10000000)
   # params = model.get_parameters()
   # print(params)
    model.train(10000000)
    model.train(10000000)
    model.train(10000000)
    model.train(10000000)
    model.train(10000000)
    model.train(10000000)
    model.set_parameters(params)
    params = model.get_parameters()
    print(params)
    model.train(10000000)
    model.train(10000000)
    model.train(10000000)
    model.train(10000000)
    
   # model.train(100000000)
   # model.train(100000000)
   # model.train(100000000)
   # model.train(20000000)
   # model.train(100000000)
   # model.train(100000000)
   # model.train(20000000)
   # model.train(20000000)

if __name__ == '__main__':
    since = time.time()
    main()
    time_elapsed = time.time()-since
    print("Total Run Time: {}".format(time_elapsed))
