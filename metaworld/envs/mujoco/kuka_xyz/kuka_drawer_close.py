import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.kuka_xyz.base import KukaXYZEnv, _assert_task_is_set


class KukaDrawerCloseEnv(KukaXYZEnv):
    '''
    Drawer
    '''
    def __init__(self):

        hand_low = (-0.5, 0.40, 0.01)
        hand_high = (0.5, 0.85, 0.5)
        # obj_low = (-0.1, 0.85, 0.04) # use this to control obj pos + goal
        # obj_high = (0.1, 0.85, 0.04)
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
            # 'obj_init_pos': np.array([0., 0.85, 0.04], dtype=np.float32),
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
        # return get_asset_full_path('kuka_xyz/kuka_drawer_v2.xml')
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
        info = {'reachDist': reachDist, 'goalDist': pullDist, 'epRew' : reward, 'pickRew':None, 'success': float(pullDist <= 0.06)}

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('handle')

    def _set_goal_marker(self, goal):
        # self.data.site_xpos[self.model.site_name2id('goal')] = (
        #     goal[:3]
        # )
        self.data.site_xpos[self.model.site_name2id('goal_drawer')] = (
            goal[:3]
        )

    def _set_obj_xyz(self, pos):
        # qpos 0-6 joints of KUKA, 7-8 gripper1, 7-14 gripper2f85, others - joints in scene
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        # qpos[9] = pos
        qpos[15] = pos
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        # self._state_goal = self.obj_init_pos - np.array([.0, .2, .0])
        self._state_goal = self.obj_init_pos + np.array([.27, .0, .0]) # switch to sim x-axis
        self.objHeight = self.data.get_geom_xpos('handle')[2]

        if self.random_init:
            obj_pos = self._get_state_rand_vec()
            self.obj_init_pos = obj_pos
            goal_pos = obj_pos.copy()
            # goal_pos[1] -= 0.2
            goal_pos[0] += 0.27
            self._state_goal = goal_pos

        self._set_goal_marker(self._state_goal)
        drawer_cover_pos = self.obj_init_pos.copy()
        drawer_cover_pos[2] -= 0.04
        self.sim.model.body_pos[self.model.body_name2id('drawer')] = self.obj_init_pos
        self.sim.model.body_pos[self.model.body_name2id('drawer_cover')] = drawer_cover_pos
        # self.sim.model.site_pos[self.model.site_name2id('goal')] = self._state_goal
        self.sim.model.site_pos[self.model.site_name2id('goal_drawer')] = self._state_goal
        self._set_obj_xyz(-0.2)
        # self.maxDist = np.abs(self.data.get_geom_xpos('handle')[1] - self._state_goal[1])
        self.maxDist = np.abs(self.data.get_geom_xpos('handle')[0] - self._state_goal[0])
        self.target_reward = 1000*self.maxDist + 1000*2

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(50):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            # self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            # reset kuka hand pose
            self.data.set_mocap_quat('mocap', np.array([0, -1, 1, 0]))
            # self.do_simulation([-1,1], self.frame_skip)
            self.do_simulation(255, self.frame_skip) # for robotiq
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2

    def compute_reward(self, actions, obs):
        del actions

        obs = obs['state_observation']

        objPos = obs[3:6]

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        # pullGoal = self._state_goal[1]
        pullGoal = self._state_goal[0]

        reachDist = np.linalg.norm(objPos - fingerCOM)

        # pullDist = np.abs(objPos[1] - pullGoal)
        pullDist = np.abs(objPos[0] - pullGoal)

        c1 = 1000
        c2 = 0.01
        c3 = 0.001

        if reachDist < 0.05:
            pullRew = 1000*(self.maxDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
            pullRew = max(pullRew, 0)
        else:
            pullRew = 0

        reward = -reachDist + pullRew

        return [reward, reachDist, pullDist]
