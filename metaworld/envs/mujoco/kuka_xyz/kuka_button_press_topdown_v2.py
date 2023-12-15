import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.kuka_xyz.base import KukaXYZEnv, _assert_task_is_set


class KukaButtonPressTopdownEnvV2(KukaXYZEnv):

    def __init__(self):

        hand_low = (-0.5, 0.40, 0.01)
        hand_high = (0.5, 0.85, 0.5)
        obj_low = (0.29, 0.64, 0.115) 
        obj_high = (0.3, 0.65, 0.115) 


        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0.29, 0.64, 0.115], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.5, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0, 0.75, 0.1]) # state goal pos = hole
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self.max_path_length = 200

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return get_asset_full_path('kuka_xyz/kuka_button_press_topdown.xml')
        # return get_asset_full_path('kuka_xyz/kuka_sequence.xml')

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
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward, reachDist, pressDist = self.compute_reward(action, obs_dict)
        self.curr_path_length +=1
        info = {'reachDist': reachDist, 'goalDist': pressDist, 
                'epRew': reward, 'pickRew':None, 'success': float(pressDist <= 0.02)}
        info['goal'] = self.goal

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.get_body_com("button")+np.array([0.0, 0.0, 0.193])

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        self._state_goal = self.goal.copy()

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self.obj_init_pos = goal_pos

        self.sim.model.body_pos[self.model.body_name2id('box')] = self.obj_init_pos
        self._state_goal = self.get_site_pos('hole')
        self.maxDist = np.abs(self.data.site_xpos[self.model.site_name2id('buttonStart')][2] - self._state_goal[2])
        self.target_reward = 1000*self.maxDist + 1000*2
        # print('button: ', self.get_body_com('button'))
        # print('buttonStart: ', self.get_site_pos('buttonStart'))
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(50):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            # reset pose
            # self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            # self.data.set_mocap_quat('mocap', np.array([0, 1, 0, 0]))
            self.data.set_mocap_quat('mocap', np.array([0, -1, 1, 0]))
            self.do_simulation(255, self.frame_skip)

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def compute_reward(self, actions, obs):
        del actions

        obs = obs['state_observation']
        objPos = obs[3:6]

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        pressGoal = self._state_goal[2]

        pressDist = np.abs(objPos[2] - pressGoal)
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
