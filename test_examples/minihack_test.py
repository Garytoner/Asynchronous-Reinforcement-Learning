import argparse
from distutils.util import strtobool
import minihack
from Asynchronous_Reinforcement_Learning.algorithms.appo.appo import APPO ,A3C,IMPALA
import torch
import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
    args.env_id ="MiniHack-River-v0"
    args.encoder_subtype ="convnet_test"
    args.num_workers = 2
    args.num_envs_per_worker = 2
    model = APPO(env=args.env_id, device=device, num_workers=args.num_workers,num_envs_per_worker=args.num_envs_per_worker,encoder=args.encoder_type,encodersubtype=args.encoder_subtype,policy_kwargs = {"num_policies":1,"with_pbt":False,"worker_num_splits":2,"batch_size":512,"encoder_custom":'minihack_encoder'})
    model.train(100000)

   
  

if __name__ == '__main__':
    since = time.time()
    main()
    time_elapsed = time.time()-since
    print("Total Run Time: {}".format(time_elapsed))
