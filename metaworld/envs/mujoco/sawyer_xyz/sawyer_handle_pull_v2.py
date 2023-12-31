import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv, _assert_task_is_set


class SawyerHandlePullEnvV2(SawyerXYZEnv):
    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.8, -0.001)
        obj_high = (0.1, 0.9, +0.001)
        goal_low = (-0.1, 0.55, 0.04)
        goal_high = (0.1, 0.70, 0.18)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.9, 0.0]),
            'hand_init_pos': np.array((0, 0.6, 0.2),),
        }
        self.goal = np.array([0, 0.8, 0.14])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.max_path_length = 150

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_handle_press.xml', True)

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward, reachDist, pressDist = self.compute_reward(action, obs_dict)
        self.curr_path_length += 1

        info = {
            'reachDist': reachDist,
            'goalDist': pressDist,
            'epRew': reward,
            'pickRew': None,
            'success': float(pressDist <= 0.04),
            'goal': self.goal
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.get_site_pos('handleStart')

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        self.obj_init_pos = (self._get_state_rand_vec()
                             if self.random_init
                             else self.init_config['obj_init_pos'])

        self.sim.model.body_pos[self.model.body_name2id('box')] = self.obj_init_pos
        self._set_obj_xyz(-0.1)
        self._state_goal = self.get_site_pos('goalPull')
        self.maxDist = np.abs(self.data.site_xpos[self.model.site_name2id('handleStart')][-1] - self._state_goal[-1])
        self.target_reward = 1000*self.maxDist + 1000*2

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(50):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)

    def compute_reward(self, actions, obs):
        del actions

        obs = obs['state_observation']

        objPos = obs[3:6]

        leftFinger = self.get_site_pos('leftEndEffector')
        fingerCOM  =  leftFinger

        pressGoal = self._state_goal[-1]

        pressDist = np.abs(objPos[-1] - pressGoal)
        reachDist = np.linalg.norm(objPos - fingerCOM)
        reachRew = -reachDist

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        if reachDist < 0.05:
            pressRew = 1000*(self.maxDist - pressDist) + c1*(np.exp(-(pressDist**2)/c2) + np.exp(-(pressDist**2)/c3))
        else:
            pressRew = 0
        pressRew = max(pressRew, 0)
        reward = reachRew + pressRew

        return [reward, reachDist, pressDist]
