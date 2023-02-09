> Please check the following link for [Custom model template](https://www.samplefactory.dev/03-customization/custom-models/#custom-model-template)

## The model template

```python
from sample_factory.model.encoder import Encoder
from sample_factory.model.decoder import Decoder
from sample_factory.model.core import ModelCore
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.algo.utils.context import global_model_factory


class CustomEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        # build custom encoder architecture
        ...

    def forward(self, obs_dict):
        # custom forward logic
        ...

class CustomCore(ModelCore):
    def __init__(self, cfg: Config, input_size: int):
        super().__init__(cfg)
        # build custom core architecture
        ...

    def forward(self, head_output, rnn_states):
        # custom forward logic
        ...


class CustomDecoder(Decoder):
    def __init__(self, cfg: Config, decoder_input_size: int):
        super().__init__(cfg)
        # build custom decoder architecture
        ...

    def forward(self, core_output):
        # custom forward logic
        ...

class CustomActorCritic(ActorCritic):
    def __init__(
        self,
        model_factory,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        cfg: Config,
    ):
    super().__init__(obs_space, action_space, cfg)

    self.encoder = CustomEncoder(cfg, obs_space)
    self.core = CustomCore(cfg, self.encoder.get_out_size())
    self.decoder = CustomDecoder(cfg, self.core.get_out_size())
    self.critic_linear = nn.Linear(self.decoder.get_out_size())
    self.action_parameterization = self.get_action_parameterization(
        self.decoder.get_out_size()
    ) 

    def forward(self, normalized_obs_dict, rnn_states, values_only=False):
        # forward logic
        ...


def register_model_components():
    # register custom components with the factory
    # you can register an entire Actor Critic model
    global_model_factory().register_actor_critic_factory(CustomActorCritic)

    # or individual components
    global_model_factory().register_encoder_factory(CustomEncoder)
    global_model_factory().register_core_factory(CustomCore)
    global_model_factory().register_decoder_factory(CustomDecoder)

def main():
    """Script entry point."""
    register_model_components()
    cfg = parse_args()

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
```

Let's took `dmlab` as an example, in [`./sample_factory/envs/dmlab/dmlab_model.py`](sample_factory/envs/dmlab/dmlab_model.py) 


- It should be noted, in this version of `SampleFactory`, envs are registered by using seperated `make_env` function, for example.

```python
def maze_funcs():
    from sample_factory.envs.maze.maze_utils import make_maze_env
    from sample_factory.envs.maze.maze_params import maze_override_defaults
    return make_maze_env, None, maze_override_defaults
```
and in `register_default_envs(env_registry)`


EnvRegistry.registry[env_name_prefix] can return


A standard thing to do in RL frameworks is to just rely on unique environment names registered in Gym.

SampleFactory supports a mechanism on top of that, we define "environment families", e.g. "atari", or "doom", and certain things can be defined per env family rather than for specific environment or experiment (such asdefault hyperparameters and env command line arguments).

For every supported family of environments we require four components:

**:param env_name_prefix:** name prefix, e.g. atari_. This allows us to register a single entry per env family rather than individual env. Prefix can also, of course, be a full name of the environment.

**:param make_env_func:** Factory function that creates an environment instance.
This function is called like:
`make_my_env(full_env_name, cfg=cfg, env_config=env_config)`
Where full_env_name is a name of the environment to be created, cfg is a namespace with all CLI arguments, and env_config is an auxiliary dictionary containing information such as worker index on which the environment lives (some envs may require this information)

**:param add_extra_params_func: (optional)** function that adds additional parameters to the argument parser.

This is a very easy way to make your envs configurable through command-line interface.

**:param override_default_params_func: (optional)** function that can override the default command line arguments in the parser. Every environment demands its own unique set of model architectures and hyperparameters, so this mechanism allows us to specify these default parameters once per family of envs to avoid typing them every time we want to launch an experiment.

See the sample_factory_examples for the default envs, it's actually very simple.

If you want to use a Gym env, just create an empty make_env_func that ignores other parameters and instantiates a copy of your Gym environment.