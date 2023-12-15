import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.kuka_xyz.base import KukaXYZEnv, _assert_task_is_set


class KukaDrawerOpenEnv(KukaXYZEnv):
    def __init__(self):

        hand_low = (-0.5, 0.40, 0.01)
        hand_high = (0.5, 0.85, 0.5)
        obj_low = (-0.5, 0.625, 0.04) # use this to control obj pos + goal
        obj_high = (-0.5, 0.63, 0.04)
        goal_low = (-0.1, 0.649, 0.04)
        goal_high = (0.1, 0.651, 0.04)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([-0.5, 0.63, 0.04], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.5, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.max_path_length = 200

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        # return get_asset_full_path('sawyer_xyz/sawyer_drawer_v2.xml')
        return get_asset_full_path('kuka_xyz/kuka_sequence.xml')

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        ##########################################################################
        # TODO: for Robotiq action must be rescaled between [-1, 1] --> [0, 255] #
        ##########################################################################
        gripper_action = self.rescale_gripper_action(action[-1])
        self.do_simulation(gripper_action)
        # self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward, reachDist, pullDist = self.compute_reward(action, obs_dict)
        self.curr_path_length +=1
        info = {'reachDist': reachDist, 'goalDist': pullDist, 'epRew' : reward, 'pickRew':None, 'success': float(pullDist <= 0.08)}

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('handle').copy()

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict['state_achieved_goal'] = (self.get_site_pos('handleStart').copy() + self.data.get_geom_xpos('drawer_wall2').copy()) / 2
        return obs_dict

    def _set_goal_marker(self, goal):
        self.data.site_xpos[self.model.site_name2id('goal_drawer')] = (
            goal[:3]
        )

    def reset_model(self):
        self._reset_hand()
        # self._state_goal = self.obj_init_pos - np.array([.0, .35, .0])
        self._state_goal = self.obj_init_pos + np.array([.27, .0, .0]) + np.array([.2, .0, -0.01]) # switch to sim x-axis
        self.objHeight = self.data.get_geom_xpos('handle')[2]

        if self.random_init:
            obj_pos = self._get_state_rand_vec()
            self.obj_init_pos = obj_pos
            goal_pos = obj_pos.copy()
            # goal_pos[1] -= 0.35
            goal_pos[0] += 0.47
            self._state_goal = goal_pos

        self._set_goal_marker(self._state_goal)
        drawer_cover_pos = self.obj_init_pos.copy()
        drawer_cover_pos[2] -= 0.04
        self.sim.model.body_pos[self.model.body_name2id('drawer')] = self.obj_init_pos
        self.sim.model.body_pos[self.model.body_name2id('drawer_cover')] = drawer_cover_pos
        self.sim.model.site_pos[self.model.site_name2id('goal_drawer')] = self._state_goal
        self.maxPullDist = 0.2
        self.target_reward = 1000*self.maxPullDist + 1000*2

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(50):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([0, -1, 1, 0]))
            # self.do_simulation([-1,1], self.frame_skip)
            self.do_simulation(255, self.frame_skip) # for robotiq

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        del actions

        obs = obs['state_observation']

        objPos = obs[3:6]
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2
        pullGoal = self._state_goal
        pullDist = np.abs(objPos[0] - pullGoal[0]) # switch to x-axis (horizontal)
        reachDist = np.linalg.norm(objPos - fingerCOM)
        reachRew = -reachDist

        self.reachCompleted = reachDist < 0.05

        def pullReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            if self.reachCompleted:
                pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
                pullRew = max(pullRew,0)
                return pullRew
            else:
                return 0



        pullRew = pullReward()
        reward = reachRew + pullRew

        return [reward, reachDist, pullDist]
