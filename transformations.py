import itertools
from typing import Sequence

import numpy as np


class Transformations:
    """

    """
    camera_resolution = np.array((4056, 3040))  # (width, height) px
    camera_to_wall_distance = 4.  # m
    _chip_size = np.array((7.564, 5.476))  # (width, height) mm
    _crop_factor = 5.54
    _lens_focal_length = np.array((2.8, 12))  # mm
    _wall_diagonal_pixels = np.sqrt(np.sum(np.square(camera_resolution)))

    def __init__(self):
        self.default_aov, self.default_fov = self.get_fov()
        self.default_ppm = (self.get_ppm(self.default_fov[0], 0), self.get_ppm(self.default_fov[1], 1))

    def get_fov(self, c2w_distance=camera_to_wall_distance, focal_length=_lens_focal_length[0]):
        """ calculate camera fov based on camera to wall distance and lens adjusted focal length

        :param c2w_distance: (float) distance from camera to projection wall in [m]
        :param focal_length: (float) lens focal length in [mm] (not adjusted to crop factor)
        :return aov, fov: Tuple[float], Tuple[float] Width x Height angle of view [deg] and field of view of the camera [m]
        """

        aov = [2*np.arctan(s/(2*focal_length)) for s in self._chip_size]  # angle of view (rad)
        fov = [2*c2w_distance*np.tan(a/2) for a in aov]  # field of view

        return tuple(np.rad2deg(aov)), tuple(fov)

    def get_ppm(self, fov: float, axis=0):
        """

        :param fov: (float) field of view in meters
        :param axis: (int) axis for which ppm is calculated | Horizontal (0) | Vertical (1) |
        :return ppm: (float) pixels per meter
        """

        resolution = self.camera_resolution[int(axis)]

        ppm = resolution/fov

        return ppm

    def meter_to_pixel(self, pos_meters: float, ppm: float, axis: int):
        """ What the camera sees

        :param pos_meters: (float) position of laser point from center in [meters]
        :param ppm: (float) number of pixels per one meter
        :param axis: (int) axis for which conversion is calculated | Horizontal (0) | Vertical (1) |
        :return pos_pixels: (int) position of laser point from center in [pixels]
        """
        pixel_position = self.camera_resolution[axis]/2 + ppm * pos_meters

        # ensure resolution boundaries!
        pixel_position = max(min(pixel_position, self.camera_resolution[axis]), 0)

        return pixel_position

    def pixel_to_meter(self, pos_pixels: int, ppm: float, axis: int):
        """

        :param pos_pixels: (int) position of laser point from center in [pixels]
        :param ppm: (float) number of pixels per one meter
        :param axis: (int) axis for which conversion is calculated | Horizontal (0) | Vertical (1) |
        :return pos_meters: (float) position of laser point from center in [meters]
        """
        return (pos_pixels-self.camera_resolution[axis]/2)/ppm

    def angle_to_meter(self, angle, angle2=90, angle_defaults=(90, 90)):
        """ convert servo angle to meter distance from _default_angle position ("center")

        :param angle: (int) current angle of the servo
        :param angle2: (int) angle of the secondary axis (if 90, it has no effect) [degrees]
        :param angle_defaults: (Tuple[int]) default angles of both axes [degrees]
        :return meter_pos: (float) distance of laser point from "center" in meters
        """
        wall_dist = self.camera_to_wall_distance
        alpha = (angle_defaults[0] - angle)/180*np.pi  # converted to rad
        beta = (angle_defaults[1] - angle2)/180*np.pi  # converted to rad
        return wall_dist*np.tan(alpha)/np.cos(beta)

    def meter_to_angle(self, dist_meters, angle2=90, angle_defaults=(90, 90)):
        """ convert meter position on wall to angle distance from _default_angle position ("center")

        :param dist_meters: (float) distance of laser point from "center" in meters
        :param angle2: (int) angle of the secondary axis (if 90, it has no effect) [degrees]
        :param angle_defaults: (Tuple[int]) default angles of both axes [degrees]
        :return angle: (int) respective angle of the servo
        """
        wall_dist = self.camera_to_wall_distance  # [meters]
        beta = np.deg2rad(angle_defaults[1] - angle2)  # [radians]
        return int(np.round(angle_defaults[0] - np.rad2deg(np.arctan(dist_meters*np.cos(beta)/wall_dist))))

    def angle_to_pixel(self, angles: Sequence[int], angle_defaults=(90, 90), ppm=(None, None)):
        """
        :param angles: Tuple[int] x and y angles of the servos for respective laser
        :param angle_defaults: Tuple[int] default values for x and y angles of the servos

        :return: Tuple[int]
        """
        assert len(angles) == 2, "angles must hold exactly 2 values"
        assert len(angle_defaults) == 2, "angle defaults must hold exactly 2 values"

#        _, fov = self.get_fov()
        ppm = list(ppm)
        if ppm[0] is None:
            ppm[0] = self.default_ppm[0]
        if ppm[1] is None:
            ppm[1] = self.default_ppm[1]
#        ppm_x = self.get_ppm(fov[0], axis=0)
#        ppm_y = self.get_ppm(fov[1], axis=1)

        m_x = self.angle_to_meter(angles[0], angles[1], angle_defaults)
        m_y = self.angle_to_meter(angles[1], angles[0], angle_defaults[::-1])

        pixel_x = self.meter_to_pixel(m_x, ppm[0], axis=0)
        pixel_y = self.meter_to_pixel(m_y, ppm[1], axis=1)

        return np.array(pixel_x, pixel_y)

    def normalize_obs(self, obs):
        obs[0] /= self.camera_resolution[0]
        obs[2] /= self.camera_resolution[0]
        obs[1] /= self.camera_resolution[1]
        obs[3] /= self.camera_resolution[1]
        return obs

    def denormalize_obs(self, norm_obs):
        norm_obs[0] *= self.camera_resolution[0]
        norm_obs[2] *= self.camera_resolution[0]
        norm_obs[1] *= self.camera_resolution[1]
        norm_obs[3] *= self.camera_resolution[1]
        return norm_obs

    @staticmethod
    def velocity(dx, dy):
        """

        :param dx: (float) angle change in x axis (normalized to -1;1)
        :param dy: (float) angle change in y axis (normalized to -1;1)
        :return: punishment for moving too much (normalized to 0; 1)
        """
        return (dx**2 + dy**2)/2.

    @staticmethod
    def distance(rx, ry, gx, gy):
        """

        euclidean distance function """
        # print(f"red: {pred_batch}")
        # print(f"grn: {target_batch}")
        dist = np.sqrt(np.square(gx - rx) + np.square(gy - ry))
        dist_max = np.sqrt(2.)
        return np.squeeze(dist/dist_max)

    @staticmethod
    def bound_proximity(x, y):
        return (max(x, (1-x))**1024 + max(y, (1-y))**1024)/2

    def reward(self, obs, action, implementation):
        """
        :param obs: (float) [gx, gy, rx, ry](t) observations in current step
        :param action: (float) [dx, dy](t) actions taken in current step (-1;1)
        :param implementation: (float) implementation of the reward function
            0 ... base reward
            1 ... sigmoid reward
            2 ... exponential reward
            3 ... exponential reward with boundary punishment
            4 ... exponential reward with velocity and boundary punishment
        :return: reward
        """
        rx0 = obs[0]
        ry0 = obs[1]
        gx0 = obs[2]
        gy0 = obs[3]
        dx = action[0]
        dy = action[1]
        dist = self.distance(rx0, ry0, gx0, gy0)
        if implementation == 0:  # base reward
            return 1. - dist
        elif implementation == 1:  # sigmoid reward
            return 1/np.exp(dist**2)**4
        elif implementation == 2:  # exponential reward
            return 1 - dist**0.4
        elif implementation == 3:  # exponential reward with boundary punishment
            dist_reward = 1 - dist**0.4
            return dist_reward - self.bound_proximity(gx0, gy0)
        elif implementation == 4:  # exponential reward with velocity discount and boundary punishment
            dist_reward = 1 - dist**0.4
            velocity_discount = (1 - self.velocity(dx, dy))**(1/max(dist, 0.2))
            return dist_reward*velocity_discount - self.bound_proximity(gx0, gy0)


class PathGenerator:

    def __init__(self, **kwargs):
        self.env = Transformations()
        self.resolution = tuple(kwargs["resolution"]) if "resolution" in kwargs.keys() else Transformations.camera_resolution
        self._initial_angles = tuple(kwargs["initial_angles"]) if "initial_angles" in kwargs.keys() else (90, 90)
        self.seed = int(kwargs["seed"]) if "seed" in kwargs.keys() else None

        self._last_angles = self._initial_angles
        self.rng = np.random.default_rng(self.seed)

    @property
    def initial_angles(self):
        return self._initial_angles

    @initial_angles.setter
    def initial_angles(self, value: tuple):
        assert type(value) == Sequence
        assert len(value) == 2
        if self._last_angles == self._initial_angles:
            self._last_angles = tuple(value)
        self._initial_angles = tuple(value)

    def random_gen(self, angle_bounds: tuple, max_angle_step: int, return_angles=False):
        assert len(angle_bounds) == 2
        a_min = angle_bounds[0]
        a_max = angle_bounds[1]

        while True:
            angles = self._last_angles + self.rng.uniform(-1, 1, 2)*max_angle_step
            angles = [min(a_max, max(a_min, a)) for a in angles]

            self._last_angles = angles

            if return_angles:
                yield angles
            else:
                yield self.env.angle_to_pixel(angles)
        # angles = angle_bounds[0] + self.rng.random(2)*(angle_bounds[1] - angle_bounds[0])

    def ellipse(self, scale=0.5, resolution=0.1*np.pi, circle=False, return_angles=False):
        """ generate pixel or servo angle points, which result in elliptical shape

        :param scale: (float) [0 to 1] size of the shape with respect to the size of working area
        :param resolution: (float) period (distance) of points in the shape
        :param circle: (bool) if True, calculate shape to be a circle | else it's an ellipse
        :param return_angles: if True, return angles of servos | else return pixel positions
        :return pixel positions/angles of ellipse/circle points
        """
        center = [r//2 for r in self.resolution]
        radius = [(r - c)*scale for r, c in zip(self.resolution, center)]

        alpha = np.arange(0, 2*np.pi, resolution)

        if circle:
            radius = [min(radius)]*2  # pick smaller radius to make a circle

        x = center[0] + radius[0]*np.cos(alpha)
        y = center[1] + radius[1]*np.sin(alpha)

        if return_angles:
            x, y = self._generate_ellipse_angles(x, y)

        return itertools.cycle(zip(x, y))

    def _generate_ellipse_angles(self, xs, ys):
        # convert pixel path to servo angles
        aov, fov = self.env.get_fov()
        ppm_x = self.env.get_ppm(fov[1], axis=1)
        ppm_y = self.env.get_ppm(fov[0], axis=0)

        xas, yas = list(), list()
        ya = 90
        for x, y in zip(xs, ys):
            xm = self.env.pixel_to_meter(x, ppm_x, axis=0)
            ym = self.env.pixel_to_meter(y, ppm_y, axis=1)
            xa = self.env.meter_to_angle(xm, ya)
            ya = self.env.meter_to_angle(ym, xa)
            xas.append(xa)
            yas.append(ya)

        return xas, yas

        # DONE: calculate points of a circle (goniometry)
        # DONE: make function for pixel->angle / angle->pixel position conversion