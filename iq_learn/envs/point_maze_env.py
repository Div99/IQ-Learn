import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py
# from mujoco_py.mjlib import mjlib
import os

import utils.logger as logger

from envs.dynamic_mjc.mjc_models import point_mass_maze

# set logger
# log = logger.setup_logger(os.path.join('training.log'))


class PointMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, direction=1, maze_length=0.6, sparse_reward=False, no_reward=False, discrete=True, episode_length=100):
        utils.EzPickle.__init__(self)
        self.sparse_reward = sparse_reward
        self.no_reward = no_reward
        self.max_episode_length = episode_length
        self.direction = direction
        self.length = maze_length
        self.discrete = discrete  # if use discrete initial positions
        self.episode_length = 0
        self.policy_contexts = None

        model = point_mass_maze(direction=self.direction, length=self.length)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

    def step(self, a):
        vec_dist = self.get_body_com("particle") - self.get_body_com("target")

        reward_dist = - np.linalg.norm(vec_dist)  # particle to target
        reward_ctrl = - np.square(a).sum()
        if self.no_reward:
            reward = 0
        elif self.sparse_reward:
            if reward_dist <= 0.1:
                reward = 1
            else:
                reward = 0
        else:
            reward = reward_dist + 0.001 * reward_ctrl

        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        self.episode_length += 1
        done = self.episode_length >= self.max_episode_length
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self, reset_args=None, policy_contexts=None):
        gaussian_mean_list = [0.1, 0.3, 0.5]
        if reset_args is None:
            target_pos = [0., 0.5, 0.]
            if self.discrete:
                target_pos_x = np.random.choice(np.arange(0., 0.5, 0.04))
                target_pos[0] = target_pos_x
            else:
                while True:
                    target_pos_x = np.random.normal(
                        loc=np.random.choice(gaussian_mean_list), scale=0.05)
                    if target_pos_x >= 0. and target_pos_x <= 0.6:
                        target_pos[0] = target_pos_x
                        break
        else:
            target_pos = reset_args
        self.policy_contexts = policy_contexts
        qpos = self.init_qpos.copy()
        body_pos = self.model.body_pos.copy()
        body_pos[2] = target_pos
        self.model.body_pos[:] = body_pos

        self.episode_length = 0
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        qpos = qpos + self.np_random.uniform(size=self.model.nq, low=-0.05, high=0.05)
        self.set_state(qpos, qvel)
        self.episode_length = 0
        return self.get_obs()

    def reset(self, reset_args=None, policy_contexts=None):
        return self._reset(reset_args=reset_args, policy_contexts=policy_contexts)

    def _reset(self, reset_args=None, policy_contexts=None):
        # original mujoco reset
        self.sim.reset()
        ob = self.reset_model(reset_args=reset_args, policy_contexts=policy_contexts)
        # if self.viewer is not None:
        #     self.viewer.autoscale()
        #     self.viewer_setup()
        return ob

    def get_obs(self):
        if self.policy_contexts is not None:
            return np.concatenate([
                self.get_body_com("particle"),
                self.policy_contexts
            ])
        return np.concatenate([
            self.get_body_com("particle"),
            self.get_body_com("target")
        ])

    def make_state(self, state):
        particle_pos, target_pos = state[:3], state[3:]
        qpos = self.init_qpos.copy()
        body_pos = self.model.body_pos.copy()

        body_pos[1] = particle_pos
        body_pos[2] = target_pos
        self.model.body_pos[:] = body_pos

        self.episode_length = 0
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        qpos = qpos + self.np_random.uniform(size=self.model.nq, low=-0.05, high=0.05)
        self.set_state(qpos, qvel)
        return self.get_obs()

    def plot_trajs(self, *args, **kwargs):
        pass

    def log_diagnostics(self, paths):
        rew_dist = np.array([traj['env_infos']['reward_dist'] for traj in paths])
        rew_ctrl = np.array([traj['env_infos']['reward_ctrl'] for traj in paths])

        # logger.info('AvgObjectToGoalDist', -np.mean(rew_dist.mean()))
        # logger.info('AvgControlCost', -np.mean(rew_ctrl.mean()))
        # logger.info('AvgMinToGoalDist', np.mean(np.min(-rew_dist, axis=1)))
