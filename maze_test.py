import argparse
import os
from distutils.util import strtobool
from sample_factory.algorithms.appo.appo import APPO,A3C,IMPALA
import torch
import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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
    parser.add_argument("--env-id", type=str, default="mujoco_CartPolev1",
        help="the id of the environment")
    parser.add_argument("--num_workers", type=int, default=2,
        help="the id of the environment")
    parser.add_argument(
            '--num_envs_per_worker', default=2, type=int,
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
    args.env_id = "maze-random-10x10-plus-v0"

    model = APPO(env=args.env_id, device=device, num_workers=args.num_workers,num_envs_per_worker=args.num_envs_per_worker,encoder=args.encoder_type,encodersubtype=args.encoder_subtype,policy_kwargs = {"num_policies":1,"reward_scale":0.01,"kl_loss_coeff":1.0,"actor_critic_share_weights":False})
    
    model.train(train_for_env_steps=10000)

if __name__ == '__main__':
    since = time.time()
    main()
    time_elapsed = time.time()-since
    print("Total Run Time: {}".format(time_elapsed))
