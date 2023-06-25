import numpy as np
from gym.spaces import  Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.kuka_xyz.base import KukaXYZEnv, _assert_task_is_set


class KukaWindowOpenEnvV2(KukaXYZEnv):
    """
    Motivation for V2:
        When V1 scripted policy failed, it was often due to limited path length.
    Changelog from V1 to V2:
        - (8/11/20) Updated to Byron's XML
        - (7/7/20) Added 3 element handle position to the observation
            (for consistency with other environments)
        - (6/15/20) Increased max_path_length from 150 to 200
    """
    def __init__(self):

        liftThresh = 0.02
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.7, 0.16)
        obj_high = (0.1, 0.9, 0.16)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([-0.1, 0.785, 0.16], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.4, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self.max_path_length = 200
        self.liftThresh = liftThresh

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.maxPullDist = 0.2
        self.target_reward = 1000 * self.maxPullDist + 1000 * 2

    @property
    def model_name(self):
        return get_asset_full_path('kuka_xyz/kuka_window_horizontal.xml', True)

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward, reachDist, pickrew, pullDist = self.compute_reward(action, obs_dict)
        self.curr_path_length += 1

        info = {
            'reachDist': reachDist,
            'goalDist': pullDist,
            'epRew': reward,
            'pickRew': pickrew,
            'success': float(pullDist <= 0.05),
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.get_site_pos('handleOpenStart')

    def _set_goal_marker(self, goal):
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )

    def reset_model(self):
        self._reset_hand()

        if self.random_init:
            self.obj_init_pos = self._get_state_rand_vec()

        self._state_goal = self.obj_init_pos + np.array([.2, .0, .0])
        self._set_goal_marker(self._state_goal)

        self.sim.model.body_pos[self.model.body_name2id(
            'window'
        )] = self.obj_init_pos
        self.data.set_joint_qpos('window_slide', 0.0)

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(50):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            # reset kuka pose
            # self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.data.set_mocap_quat('mocap', np.array([0, 1, 0, 0]))
            
            self.do_simulation([-1, 1], self.frame_skip)

        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        del actions

        obs = obs['state_observation']

        objPos = obs[3:6]

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        pullGoal = self._state_goal

        pullDist = np.abs(objPos[0] - pullGoal[0])
        reachDist = np.linalg.norm(objPos - fingerCOM)

        self.reachCompleted = reachDist < 0.05

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        reachRew = -reachDist
        if self.reachCompleted:
            pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
        else:
            pullRew = 0
        reward = reachRew + pullRew

        return [reward, reachDist, None, pullDist]
