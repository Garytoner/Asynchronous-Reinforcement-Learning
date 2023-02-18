import torch
from torch import nn

from Asynchronous_Reinforcement_Learning.algorithms.appo.model_utils import get_obs_shape, create_standard_encoder, EncoderBase, register_custom_encoder
from Asynchronous_Reinforcement_Learning.utils.utils import log

class MazeEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        obs_shape = get_obs_shape(obs_space)
        assert len(obs_shape.obs) == 1

        if cfg.encoder_subtype == 'mlp_mujoco':
            fc_encoder_layer = cfg.hidden_size
            encoder_layers = [
                nn.Linear(obs_shape.obs[0], fc_encoder_layer),
                nn.Tanh(),
                nn.Linear(fc_encoder_layer, fc_encoder_layer),
                nn.ReLU(),
            ]
        else:
            raise NotImplementedError(f'Unknown mlp encoder {cfg.encoder_subtype}')

        self.mlp_head = nn.Sequential(*encoder_layers)
        self.init_fc_blocks(fc_encoder_layer)

    def forward(self, obs_dict):
        x = self.mlp_head(obs_dict['obs'])
        x = self.forward_fc_blocks(x)
        #print("MAZE+++ENCODER")
        return x


def maze_register_models():
    register_custom_encoder('maze_encoder', MazeEncoder)