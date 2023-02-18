

## Setup instructions

- Follow instructions in the repo to install minihcak environments

```
conda install cmake flex bison bzip2
conda install -c conda-forge cxx-compiler
install all of the sysroot libraries here: https://github.com/conda-forge/linux-sysroot-feedstock (it seems unlikely they are all necessary, but I have only tested with all)
pip install nle
pip install minihack
```

- modify class MiniHack

```
vi ./lib/python3.8/site-packages/minihack/base.py
modify class MiniHack like this:
#self.observation_space = gym.spaces.Dict(
        #    self._get_obs_space_dict(dict(NLE_SPACE_ITEMS))
        #)

        self.observation_space = self._get_obs_space_dict(dict(NLE_SPACE_ITEMS))

 def _get_obs_space_dict(self, space_dict):
        obs_space_dict = {}
        for key in self._minihack_obs_keys:
            if key in space_dict.keys():
                obs_space_dict[key] = space_dict[key]
            elif key in MINIHACK_SPACE_FUNCS.keys():
                space_func = MINIHACK_SPACE_FUNCS[key]
                obs_space_dict[key] = space_func(
                    self.obs_crop_h, self.obs_crop_w
                )
            else:
                if "pixel" in self._minihack_obs_keys:
                    d_shape = OBSERVATION_DESC["glyphs"]["shape"]
                    shape = (
                        d_shape[0] * N_TILE_PIXEL,
                        d_shape[1] * N_TILE_PIXEL,
                        3,
                    )
                    obs_space_dict["pixel"] = gym.spaces.Box(
                        low=0,
                        high=RGB_MAX_VAL,
                        shape=shape,
                        dtype=np.uint8,
                    )
                    return obs_space_dict["pixel"]
                else:
                    raise ValueError(
                        f'Observation key "{key}" is not supported'
                    )

        return obs_space_dict
        
    def _get_observation(self, observation):
        # Filter out observations that we don't need
        observation = super()._get_observation(observation)
        obs_dict = {}
        for key in self._minihack_obs_keys:
            if "pixel" in key:
                continue
            if key in self._observation_keys:
                obs_dict[key] = observation[key]
            elif key in MINIHACK_SPACE_FUNCS.keys():
                orig_key = key.replace("_crop", "")
                if "tty" in orig_key:
                    loc = observation["tty_cursor"][::-1]
                else:
                    loc = observation["blstats"][:2]
                obs_dict[key] = self._crop_observation(
                    observation[orig_key], loc
                )

        if "pixel" in self._minihack_obs_keys:
            obs_dict["pixel"] = self._glyph_mapper.to_rgb(
                observation["glyphs"]
            )
            return obs_dict["pixel"]

        if "pixel_crop" in self._minihack_obs_keys:
            obs_dict["pixel_crop"] = self._glyph_mapper.to_rgb(
                obs_dict["glyphs_crop"]
            )

        return obs_dict
```

- import minihack:

```
Do not import minihack in minihack_utils.py
import minihack in the begin of main.py of the program,in this repo is minihack_test.py      
```

- Set minibatchsize

```
set proper minibatchsize,in this repo is 512,otherwize you may encounter the cuda memory problem
```
