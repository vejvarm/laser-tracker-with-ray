import time
import numpy as np
import gym
from gym import spaces

from itertools import cycle

from environment import Laser, Wall
from helpers import generate_path
from transformations import Transformations

t = Transformations()

PIXEL_WIDTH = t.camera_resolution[0]
PIXEL_HEIGHT = t.camera_resolution[1]

ACTION_SCALE = 10.


class Trajectories:
    """
    random walk
    random circle
    """

    def __init__(self, num_steps: int, max_step_size: int, max_revolutions=5, area_width=PIXEL_WIDTH, area_height=PIXEL_HEIGHT, wall_repel_rate=0.2):
        """

        :param num_steps (int): number of steps per episode
        :param max_step_size (int): maximum size of step (in pixels)
        :param max_revolutions (int): maximum number of circle revolutions per episode
        :param area_width (int): width of working area (in pixels)
        :param area_height (int) height of working area (in pixels)
        :param wall_repel_rate (float): percentage of resolution defining minimum distance of center from area walls
        """

        self.num_steps = int(num_steps)

        self.min_step_size = 10  # px
        self.max_step_size = int(max(self.min_step_size, max_step_size))  # px

        self.max_revolutions = int(max_revolutions)

        self.min_radius = 100  # px

        # Area resolution
        self.aW = int(area_width)  # px
        self.aH = int(area_height)  # px

        # Wall repel rate for center generator and random generator
        self.wrR = min(max(0., wall_repel_rate), 1.)  # percentage
        self.wrW = int(self.wrR*self.aW)  # px
        self.wrH = int(self.wrR*self.aH)  # px

        # generators
        self._step_gen = np.random.default_rng()
        self._revolution_gen = np.random.default_rng()
        self._center_pos_gen = np.random.default_rng()
        self._radius_size_gen = np.random.default_rng()

        # TODO: define random step size wrt. max_step_size
        # TODO: define random center and radius for circle wrt. wall_repel_rate

    def _step_size(self):
        return self._step_gen.uniform(self.min_step_size, self.max_step_size)

    def _generate_circle(self, c_w, c_h, r):
        num_revolutions = self._revolution_gen.integers(1, self.max_revolutions+1)
        angles = np.linspace(0, 2*np.pi, self.num_steps//num_revolutions)

        pos_w = c_w + r*np.cos(angles)
        pos_h = c_h + r*np.sin(angles)

        return cycle(zip(pos_w, pos_h))

    def centered_circle_gen(self):
        c_w = self.aW/2
        c_h = self.aH/2

        max_radius = int(min(c_h, c_w, self.aH-c_h, self.aW-c_w))
        r = self._radius_size_gen.uniform(min(self.min_radius, max_radius), max_radius)  # circle radius (px)

        return self._generate_circle(c_w, c_h, r)

    def random_circle_gen(self):
        # generator for random circle trajectory
        c_w = self._center_pos_gen.uniform(self.wrW, self.aW-self.wrW)  # center pos. w (px)
        c_h = self._center_pos_gen.uniform(self.wrH, self.aH-self.wrH)  # center pos. h (px)

        max_radius = min(c_h, c_w, self.aH-c_h, self.aW-c_w)
        r = self._radius_size_gen.uniform(self.min_radius, max_radius)  # circle radius (px)

        # pixel_step = self._step_size()
        # angle_step = 2*np.arcsin(pixel_step/(2*r))

        return self._generate_circle(c_w, c_h, r)

    def random_walk_gen(self):
        c_w = self._center_pos_gen.uniform(self.wrW, self.aW-self.wrW)  # center pos. w (px)
        c_h = self._center_pos_gen.uniform(self.wrH, self.aH-self.wrH)  # center pos. h (px)

        raise NotImplementedError("Implement this!")


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    l_red = Laser()
    l_green = Laser()
    steps_per_episode = 200

    def __init__(self, env_config: dict):
        super(CustomEnv, self).__init__()

        # Action and observation space
        self.action_space = spaces.Box(low=-1., high=1., shape=(2, ), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0., 0., 0., 0.]),
                                            high=np.array([1., 1., 1., 1.]),
                                            dtype=np.float32)
        # image observation space
        # self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, CHANNLES), dtype=np.uint8)

        self.red_path_type = env_config["red_path_type"] if "red_path_type" in env_config.keys() else "random_circle"
        self.path_max_step_size = env_config["path_max_step_size"] if "path_max_step_size" in env_config.keys() else 100
        self.render_flag = env_config["render"] if "render" in env_config.keys() else None
        self.reward_impl = env_config["reward_impl"] if "reward_impl" in env_config.keys() else None

        if self.render_flag:
            self.wall = Wall(blit=True)
        else:
            self.wall = None

        if self.reward_impl is None:
            raise AttributeError("Must specify reward function implementation")

        self._t = Trajectories(self.steps_per_episode, self.path_max_step_size)

        if self.red_path_type == "random_circle":
            self.gen_trajectory = self._t.random_circle_gen
        elif self.red_path_type == "centered_circle":
            self.gen_trajectory = self._t.centered_circle_gen
        elif self.red_path_type == "random_walk":
            self.gen_trajectory = self._t.random_walk_gen  # TODO: implement
        else:
            raise AttributeError("'red_path_type' must be 'random_circle', 'centered_circle' or 'random_walk'")

        # initial reset of environment
        self.red_pos_gen = None
        self.reset()

        self.current_step = 1
        self.done = False
        self.reward = 0.

    def step(self, action):
        """ Executes one time-step of the environment

        :param action: proposed action from agent
        """
        if self.done:
            obs = self.reset()
            return obs, self.reward, self.done, {}

        self._update_env()

        self._take_action(action)

        self.current_step += 1

        if self.current_step >= self.steps_per_episode:
            self.done = True

        obs = self._next_observation()
        self.reward = t.reward(obs, action, self.reward_impl)

        return obs, self.reward, self.done, {}

    def _update_env(self):
        # Move target laser (red) one tick following path from init
        red_x, red_y = next(self.red_pos_gen)
        self.l_red.wall_pos_x = red_x
        self.l_red.wall_pos_y = red_y

    def _take_action(self, action):
        # Move green laser based on given action
        x_pos = self.l_green.angle_x + action[0]*ACTION_SCALE
        y_pos = self.l_green.angle_y + action[1]*ACTION_SCALE
        # print(f"before: {self.l_green.angle_x}, {self.l_green.angle_y}")
        self.l_green.move_angle_tick(x_pos, y_pos, speed_restrictions=True)
        # print(f"after: {self.l_green.angle_x}, {self.l_green.angle_y}")
        # print("\n\n#######################")

    def reset(self):
        """ Reset the environment to initial state """
        self.red_pos_gen = self.gen_trajectory()
        self.current_step = 1
        self.done = False
        self.reward = 0
        self._update_env()
        self._take_action(np.array([90., 90.]))
        return self._next_observation()

    def _next_observation(self):
        red_x = self.l_red.wall_pos_x/PIXEL_WIDTH
        red_y = self.l_red.wall_pos_y/PIXEL_HEIGHT
        green_x = self.l_green.wall_pos_x/PIXEL_WIDTH
        green_y = self.l_green.wall_pos_y/PIXEL_HEIGHT

        return np.array([red_x, red_y, green_x, green_y])

    def render(self, mode="human", close=False):
        """ Render the environment to the screen """
        if self.wall is not None:
            rx, ry, gx, gy = self._next_observation()
            rx *= PIXEL_WIDTH
            ry *= PIXEL_HEIGHT
            gx *= PIXEL_WIDTH
            gy *= PIXEL_HEIGHT
            self.wall.update((rx, ry), (gx, gy))
            time.sleep(0.05)
        else:
            print(f"Step: {self.current_step}/{self.steps_per_episode} | Reward: {self.reward}")

