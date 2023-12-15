import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.kuka_xyz.base import KukaXYZEnv, _assert_task_is_set


class KukaPickPlaceEnvV2(KukaXYZEnv):
    """
    Motivation for V2:
        V1 was very difficult to solve because the observation didn't say where
        to move after picking up the puck.
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/15/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._state_goal - pos_hand)
        - (6/15/20) Separated reach-push-pick-place into 3 separate envs.
    """
    def __init__(self):
        liftThresh = 0.04

        # goal_low = (-0.1, 0.75, 0.01) # origin 0.8
        # goal_high = (0.1, 0.82, 0.35) # origin 0.9
        goal_low = (-0.3, 0.6, 0.23)
        goal_high = (-0.29, 0.6, 0.23)
        hand_low = (-0.5, 0.4, 0.0)
        hand_high = (0.5, 0.85, 0.5) # origin 1
        obj_low = (0.05, 0.595, 0.015) # origin 0.6
        obj_high = (0.05, 0.6, 0.015) # origin 0.7


        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0, 0.6, 0.015]),
            'hand_init_pos': np.array([0, .5, .2]),
        }

        self.goal = np.array([0.1, 0.8, 0.2])

        self.obj_init_angle = self.init_config['obj_init_angle']
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh
        self.max_path_length = 200

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.num_resets = 0

    @property
    def model_name(self):
        # return get_asset_full_path('kuka_xyz/kuka_pick_place_v2.xml')
        return get_asset_full_path('kuka_xyz/kuka_sequence.xml')

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        # self.do_simulation([action[-1], -action[-1]])
        ##########################################################################
        # TODO: for Robotiq action must be rescaled between [-1, 1] --> [0, 255] #
        ##########################################################################
        gripper_action = self.rescale_gripper_action(action[-1])
        self.do_simulation(gripper_action)
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)

        ob = self._get_obs()
        obs_dict = self._get_obs_dict()

        rew, reach_dist, pick_rew, placing_dist = self.compute_reward(action, obs_dict)
        success = float(placing_dist <= 0.07)

        info = {
            'reachDist': reach_dist,
            'pickRew': pick_rew,
            'epRew': rew,
            'goalDist': placing_dist,
            'success': success,
            'goal': self.goal
        }

        self.curr_path_length += 1
        return ob, rew, False, info

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def _set_goal_marker(self, goal):
        self.data.site_xpos[self.model.site_name2id('goal_obj')] = goal[:3]

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        # qpos[9:12] = pos.copy()
        # qvel[12:15] = 0
        qpos[18:21] = pos.copy()
        qvel[-3:] = 0
        qpos[15] = -0.18 # sequence task: set drawer position as opened mode
        self.set_state(qpos, qvel)

    def fix_extreme_obj_pos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not
        # aligned. If this is not done, the object could be initialized in an
        # extreme position
        diff = self.get_body_com('obj')[:2] - \
               self.get_body_com('obj')[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0,
        # and geom_pos[2] is the object height
        return [
            adjusted_pos[0],
            adjusted_pos[1],
            self.get_body_com('obj')[-1]
        ]

    def reset_model(self):
        self._reset_hand()
        self._state_goal = self.goal.copy()
        self.obj_init_pos = self.fix_extreme_obj_pos(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.objHeight = self.get_body_com('obj')[2]
        self.heightTarget = self.objHeight + self.liftThresh

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self._state_goal = goal_pos[3:]
            while np.linalg.norm(goal_pos[:2] - self._state_goal[:2]) < 0.15:
                goal_pos = self._get_state_rand_vec()
                self._state_goal = goal_pos[3:]
            self._state_goal = goal_pos[-3:]
            self.obj_init_pos = goal_pos[:3]

        self._set_goal_marker(self._state_goal)
        self._set_obj_xyz(self.obj_init_pos)
        self.maxPlacingDist = np.linalg.norm(
            np.array([self.obj_init_pos[0],
                      self.obj_init_pos[1],
                      self.heightTarget]) -
            np.array(self._state_goal)) + self.heightTarget
        self.target_reward = 1000*self.maxPlacingDist + 1000*2
        self.num_resets += 1

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(50):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            # reset orientation to fit kuka robot
            # self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            # self.data.set_mocap_quat('mocap', np.array([0, 1, 0, 0]))
            self.data.set_mocap_quat('mocap', np.array([0, -1, 1, 0]))
            self.do_simulation(255, self.frame_skip)

        finger_right, finger_left = (
            self.get_site_pos('rightEndEffector'),
            self.get_site_pos('leftEndEffector')
        )
        self.init_finger_center = (finger_right + finger_left) / 2
        self.pick_completed = False

    def compute_reward(self, actions, obs):
        obs = obs['state_observation']
        pos_obj = obs[3:6]

        finger_right, finger_left = (
            self.get_site_pos('rightEndEffector'),
            self.get_site_pos('leftEndEffector')
        )
        finger_center = (finger_right + finger_left) / 2
        heightTarget = self.heightTarget

        goal = self._state_goal
        assert np.all(goal == self.get_site_pos('goal_obj'))

        tolerance = 0.01
        self.pick_completed = pos_obj[2] >= (heightTarget - tolerance)

        reach_dist = np.linalg.norm(finger_center - pos_obj)
        placing_dist = np.linalg.norm(pos_obj - goal)

        def obj_dropped():
            # Object on the ground, far away from the goal, and from the gripper
            # Can tweak the margin limits
            return (pos_obj[2] < (self.objHeight + 0.005))\
                   and (placing_dist > 0.02)\
                   and (reach_dist > 0.02)

        def reach_reward():
            reach_xy = np.linalg.norm(pos_obj[:-1] - finger_center[:-1])
            z_rew = np.linalg.norm(finger_center[-1] - self.init_finger_center[-1])

            reach_rew = -reach_dist if reach_xy < 0.05 else -reach_xy - 2*z_rew
            # Incentive to close fingers when reachDist is small
            if reach_dist < 0.05:
                reach_rew = -reach_dist + max(actions[-1], 0)/50

            return reach_rew, reach_dist

        def pick_reward():
            h_scale = 100
            if self.pick_completed and not obj_dropped():
                return h_scale * heightTarget
            elif (reach_dist < 0.1) and (pos_obj[2] > (self.objHeight + 0.005)):
                return h_scale * min(heightTarget, pos_obj[2])
            else:
                return 0

        def place_reward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            if self.pick_completed and reach_dist < 0.1 and not obj_dropped():
                place_rew = c1 * (self.maxPlacingDist - placing_dist) + \
                            c1 * (np.exp(-(placing_dist ** 2) / c2) +
                                  np.exp(-(placing_dist ** 2) / c3))
                place_rew = max(place_rew, 0)
                return [place_rew, placing_dist]
            else:
                return [0, placing_dist]

        reach_rew, reach_dist = reach_reward()
        pick_rew = pick_reward()
        place_rew, placing_dist = place_reward()
        assert ((place_rew >= 0) and (pick_rew >= 0))

        reward = reach_rew + pick_rew + place_rew
        return [reward, reach_dist, pick_rew, placing_dist]
