import torch
from torch import nn

from sample_factory.algorithms.appo.model_utils import get_obs_shape, create_standard_encoder, EncoderBase, register_custom_encoder,nonlinearity,fc_after_encoder_size,calc_num_elements
from sample_factory.utils.utils import log


class minihackEncoder(EncoderBase):
    class ConvEncoderImpl(nn.Module):
        """
        After we parse all the configuration and figure out the exact architecture of the model,
        we devote a separate module to it to be able to use torch.jit.script (hopefully benefit from some layer
        fusion).
        """
        def __init__(self, activation, conv_filters, fc_layer_size, encoder_extra_fc_layers, obs_shape):
            super(minihackEncoder.ConvEncoderImpl, self).__init__()
            conv_layers = []
            for layer in conv_filters:
                if layer == 'maxpool_2x2':
                    conv_layers.append(nn.MaxPool2d((2, 2)))
                elif isinstance(layer, (list, tuple)):
                    inp_ch, out_ch, filter_size, stride = layer
                    conv_layers.append(nn.Conv2d(inp_ch, out_ch, filter_size, stride=stride))
                    conv_layers.append(activation)
                else:
                    raise NotImplementedError(f'Layer {layer} not supported!')

            self.conv_head = nn.Sequential(*conv_layers)
            self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)

            fc_layers = []
            for i in range(encoder_extra_fc_layers):
                size = self.conv_head_out_size if i == 0 else fc_layer_size
                fc_layers.extend([nn.Linear(size, fc_layer_size), activation])

            self.fc_layers = nn.Sequential(*fc_layers)

        def forward(self, obs):
            x = self.conv_head(obs)
            x = x.contiguous().view(-1, self.conv_head_out_size)
            x = self.fc_layers(x)
            return x

    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        obs_shape = get_obs_shape(obs_space)
        input_ch = obs_shape.obs[0]


        if cfg.encoder_subtype == 'convnet_simple':
            conv_filters = [[input_ch, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]
        elif cfg.encoder_subtype == 'convnet_test':
            conv_filters = [[input_ch, 32, 8, 4], [32, 64, 4, 2], [64, 64, 3, 1]]
        else:
            raise NotImplementedError(f'Unknown encoder {cfg.encoder_subtype}')

        activation = nonlinearity(self.cfg)
        fc_layer_size = fc_after_encoder_size(self.cfg)
        encoder_extra_fc_layers = self.cfg.encoder_extra_fc_layers

        enc = self.ConvEncoderImpl(activation, conv_filters, fc_layer_size, encoder_extra_fc_layers, obs_shape)
        self.enc = torch.jit.script(enc)

        self.encoder_out_size = calc_num_elements(self.enc, obs_shape.obs)
        log.debug('Encoder output size: %r', self.encoder_out_size)

    def forward(self, obs_dict):
        return self.enc(obs_dict['obs'])


def minihack_register_models():
    register_custom_encoder('minihack_encoder', minihackEncoder)
