
## Setup instructions

- Follow instructions in the repo to install maze environments

```bash
git clone https://github.com/MattChanTK/gym-maze
cd gym-maze
python setup.py install
```

- unenble render

```bash
cd ./envs
vi maze_env.py
```

set enable_render to False like this:

```python
class MazeEnvRandom10x10Plus(MazeEnv):

    def __init__(self, enable_render=False):
        super(MazeEnvRandom10x10Plus, self).__init__(maze_size=(10, 10), mode="plus", enable_render=enable_render)
```

- modify move_robot function

```bash
vi modify maze_view_2d.py
```

modify move_robot function like this:

```python
 def move_robot(self, dir):
        if dir not in self.__maze.COMPASS.keys():
            if str(dir) == '0':
               dir = 'N'
            elif str(dir) == '1':
               dir = 'E'
            elif str(dir) == '2':
               dir = 'S'
            elif str(dir) == '3':
               dir = 'W'
            else:
               raise ValueError("dir cannot be %s. The only valid dirs are %s."
                             % (str(dir), str(self.__maze.COMPASS.keys())))

        if self.__maze.is_open(self.__robot, dir):

            # update the drawing
            self.__draw_robot(transparency=0)

            # move the robot
            self.__robot += np.array(self.__maze.COMPASS[dir])
            # if it's in a portal afterward
            if self.maze.is_portal(self.robot):
                self.__robot = np.array(self.maze.get_portal(tuple(self.robot)).teleport(tuple(self.robot)))
            self.__draw_robot(transparency=255)
           
```
