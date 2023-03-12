import pybullet as p
import pybullet_data
import numpy as np
import ipdb
import gym
from gym import spaces


class Panda:
    """
    Joints:
    0: joint1
    1: joint2
    2: joint3
    3: joint4
    4: joint5
    5: joint6
    6: joint7
    7: joint8 (fixed)
    8: hand_joint (fixed)
    9: finger_joint1
    10: finger_joint2
    11: grasptarget_hand (fixed)
    """

    def __init__(self, cid):
        self.cid = cid
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        self.id = p.loadURDF('franka_panda/panda.urdf', [0, 0.3, 0], useFixedBase=True, physicsClientId=self.cid)
        self._n_joints = self._n_links = p.getNumJoints(self.id, physicsClientId=self.cid)
        self._dofs = []
        self._joint_name_to_i_dof = {}
        self._joint_name_to_i_joint = {}

        self._ll = []
        self._ul = []
        self._jd = []

        j = 0
        for i in range(self._n_joints):
            joint_info = p.getJointInfo(self.id, i, physicsClientId=self.cid)

            joint_name = joint_info[1].decode()
            joint_type = joint_info[2]
            self._joint_name_to_i_joint[joint_name] = i
            if joint_type in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]:
                self._dofs.append(i)
                self._jd.append(joint_info[6])
                self._ll.append(joint_info[8])
                self._ul.append(joint_info[9])

                self._joint_name_to_i_dof[joint_name] = j
                j += 1
        self._i_ee = self._joint_name_to_i_joint['panda_grasptarget_hand']
        self._ee_ori = np.array([np.pi, 0, -np.pi / 2])

        # # make hand transparent as it gets in the way of the hand camera
        # p.changeVisualShape(
        #     self.id,
        #     self._joint_name_to_i_joint['panda_hand_joint'],
        #     rgbaColor=[1, 1, 1, 0.],
        #     physicsClientId=self.cid
        # )

        # # make fingers translucent
        # p.changeVisualShape(
        #     self.id,
        #     self._joint_name_to_i_joint['panda_finger_joint1'],
        #     rgbaColor=[1, 1, 1, 0.5],
        #     physicsClientId=self.cid
        # )
        # p.changeVisualShape(
        #     self.id,
        #     self._joint_name_to_i_joint['panda_finger_joint2'],
        #     rgbaColor=[1, 1, 1, 0.5],
        #     physicsClientId=self.cid
        # )

        # # change finger friction
        # p.changeDynamics(
        #     self.id,
        #     self._joint_name_to_i_joint['panda_finger_joint1'],
        #     lateralFriction=3.,
        #     physicsClientId=self.cid
        # )
        # p.changeDynamics(
        #     self.id,
        #     self._joint_name_to_i_joint['panda_finger_joint2'],
        #     lateralFriction=3.,
        #     physicsClientId=self.cid
        # )

        self._n_dofs = len(self._dofs)

        self._rp = [-np.pi / 2, -0.3, 0., -np.pi / 2., 0., np.pi / 2, 0.8, 0., 0.]
        self._ll[-2:] = [0., 0.]
        self._ul[-2:] = [0., 0.]
        self._jr = [7.0] * self._n_dofs

        ee_pos, _, ee_grip = self.get_ee_pos_ori_grip()
        contact_flags = self.get_contact_flags()
        self.observation_spaces = {
            'ee_pos_rel_base': spaces.Box(-np.inf, np.inf, ee_pos.shape, np.float32),
            'ee_grip': spaces.Box(0, 0.081, ee_grip.shape, np.float32),
            'contact_flags': spaces.Box(-1., 1., contact_flags.shape, np.float32)
        }
        self.action_space = spaces.Box(-1, 1, (4,), np.float32)

        self._is_closing_gripper = False # used for 'holding' detection

    def reset(self, pos):

        # IK uses current joint positions, so important to first set to rp
        for i, q in zip(self._dofs, self._rp):
            p.resetJointState(self.id, i, q, physicsClientId=self.cid)

        conf = self._inverse_kinematics(pos)
        conf[self._joint_name_to_i_dof['panda_finger_joint1']] = 0.04
        conf[self._joint_name_to_i_dof['panda_finger_joint2']] = 0.04
        for i, q in zip(self._dofs, conf):
            p.resetJointState(self.id, i, q, physicsClientId=self.cid)

        self._is_closing_gripper = False

    def _inverse_kinematics(self, pos):
        conf = np.array(p.calculateInverseKinematics(
            self.id,
            self._i_ee,
            targetPosition=pos,
            targetOrientation=p.getQuaternionFromEuler(self._ee_ori),
            lowerLimits=self._ll,
            upperLimits=self._ul,
            jointRanges=self._jr,
            restPoses=self._rp,
            maxNumIterations=100,
            residualThreshold=1e-4,
            physicsClientId=self.cid
        ))
        return conf

    def get_observation(self):
        ee_pos, _, ee_grip = self.get_ee_pos_ori_grip()
        contact_flags = self.get_contact_flags()
        obs = {
            'ee_pos_rel_base': self.get_ee_pos_rel_base(ee_pos),
            'ee_grip': ee_grip,
            'contact_flags': contact_flags
        }
        return obs

    def get_ee_pos_ori_grip(self):
        state = p.getLinkState(self.id, self._i_ee, physicsClientId=self.cid)
        pos = np.array(state[4])
        ori = np.array(p.getEulerFromQuaternion(state[5], physicsClientId=self.cid))
        states = p.getJointStates(self.id, [self._joint_name_to_i_joint['panda_finger_joint1'],
                                            self._joint_name_to_i_joint['panda_finger_joint2']],
                                  physicsClientId=self.cid)
        grip = np.array(states[0][0] + states[1][0]).reshape((1,))
        return pos, ori, grip

    def get_ee_pos_rel_base(self, ee_pos):
        base_pos, _ = p.getBasePositionAndOrientation(self.id, physicsClientId=self.cid)
        return ee_pos - base_pos

    def step(self, delta_pos, delta_grip):
        cur_pos, cur_ori, cur_grip = self.get_ee_pos_ori_grip()
        pos = cur_pos + delta_pos
        grip = cur_grip + delta_grip

        conf = self._inverse_kinematics(pos)
        conf[self._joint_name_to_i_dof['panda_finger_joint1']] = grip / 2
        conf[self._joint_name_to_i_dof['panda_finger_joint2']] = grip / 2
        p.setJointMotorControlArray(
            self.id,
            self._dofs,
            controlMode=p.POSITION_CONTROL,
            targetPositions=conf,
            targetVelocities=[0 for j in conf],
            forces=[500] * self._n_dofs,
            positionGains=[1. for j in conf],
            velocityGains=[1. for j in conf],
            physicsClientId=self.cid
        )

        # Record whether the robot's gripper is opening or closing.
        # This is important for get_is_holding().
        if delta_grip < 0:
            self._is_closing_gripper = True
        else:
            self._is_closing_gripper = False

    def get_ego_view_matrix(self):
        ee_state = p.getLinkState(self.id, self._i_ee, physicsClientId=self.cid)
        ee_pos, ee_ori = np.array(ee_state[0]), np.array(ee_state[1])
        ee_R = p.getMatrixFromQuaternion(ee_ori, physicsClientId=self.cid)
        ee_R = np.array(ee_R).reshape(3, 3)
        ee_x, ee_y, ee_z = ee_R[:, 0], ee_R[:, 1], ee_R[:, 2]
        eye = ee_pos - 0.08 * ee_z - 0.15 * ee_x
        target = ee_pos + 0.1 * ee_z
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=target,
            cameraUpVector=ee_x,
            physicsClientId=self.cid
        )
        return view_matrix

    def get_q_dq(self):
        q = []
        dq = []
        joint_states = p.getJointStates(self.id, self._dofs, physicsClientId=self.cid)
        for joint_state in joint_states:
            q.append(joint_state[0])
            dq.append(joint_state[1])

        return np.array(q), np.array(dq)

    def get_is_holding(self, id):
        fj1_contacts = p.getContactPoints(self.id, id, self._joint_name_to_i_joint['panda_finger_joint1'],
                                          physicsClientId=self.cid)
        fj2_contacts = p.getContactPoints(self.id, id, self._joint_name_to_i_joint['panda_finger_joint2'],
                                          physicsClientId=self.cid)

        # Detect whether the gripper stalled while closing.
        _, _, ee_grip = self.get_ee_pos_ori_grip()
        assert ee_grip.shape == (1,)
        ee_grip_val = ee_grip[0]
        gripper_is_slightly_open = ee_grip_val > 0.015 # 0.04-0.05 is the avg ee_grip while holding cube
        gripper_stalled_while_closing = self._is_closing_gripper and gripper_is_slightly_open
        return gripper_stalled_while_closing and len(fj1_contacts) > 0 and len(fj2_contacts) > 0

    def get_contact_flags(self):
        fj1_contacts = p.getContactPoints(self.id, linkIndexA=self._joint_name_to_i_joint['panda_finger_joint1'],
                                          physicsClientId=self.cid)
        fj2_contacts = p.getContactPoints(self.id, linkIndexA=self._joint_name_to_i_joint['panda_finger_joint2'],
                                          physicsClientId=self.cid)
        return np.array([1. if len(c) > 0 else -1. for c in [fj1_contacts, fj2_contacts]])