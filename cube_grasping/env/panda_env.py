import glob
import gym
from gym import spaces
import numpy as np
from cube_grasping.env.panda import Panda
from cube_grasping import asset
import os
import pybullet as p
import pybullet_data
import random
import time
from gym.utils import seeding
import ipdb
import pkgutil
import cv2


class PandaEnv(gym.Env):

    def __init__(self, train=True, view=1, image_obs=True, gui=False, real_time=False, seed=42,
                 random_table_texture=False, num_table_textures_train=5, num_table_textures_test=10, random_target_init_pos=True, random_panda_init_pos=True,
                 z_shift=0., add_distractors=False, distractor_color=None):
        self._gui = gui
        self._real_time = real_time
        self._render_size = [84, 84]
        self.time_step = 1. / 240
        self.action_repeat = 24
        self._cam_dist = 1.8
        self._cam_yaw = 30
        self._cam_pitch = -30
        self._cam_target_pos = [0, -0.25, 0]
        self._train = train
        self.seed(seed)
        self._view = view
        assert self._view in [1, 3, 'both'], f"Invalid arg passed in to PandaEnv constructor: view={view}" # 1: hand-centric camera, 3: third-person camera, 'both': hand-centric and third-person cameras
        self._random_table_texture = random_table_texture
        self._random_target_init_pos = random_target_init_pos
        self._random_panda_init_pos = random_panda_init_pos
        self._image_obs = image_obs
        self._add_distractors = add_distractors

        self._old_target_z_scaled = 0. # used in the reward function

        self.a_pos_scale = 0.1
        self.a_grip_scale = 0.05

        if self._gui:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, self._cam_target_pos,
                                         physicsClientId=cid)
        else:
            cid = p.connect(p.DIRECT)
            # # Enabling EGL rendering messes up the textures, so enable it only when not using table textures by
            # # uncommenting the code below (alternatively, don't enable it at all).
            # if not self._random_table_texture:
            #     egl = pkgutil.get_loader('eglRenderer')
            #     if (egl):
            #         pluginId = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin", physicsClientId=cid)
            #     else:
            #         pluginId = p.loadPlugin("eglRendererPlugin", physicsCientId=cid)
            #     print("pluginId=", pluginId)
        self.cid = cid

        self.panda = Panda(cid=cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        self.table_id = p.loadURDF(
            fileName='table/table.urdf',
            basePosition=[0, -0.3, -0.60 + z_shift],
            useFixedBase=True,
            physicsClientId=self.cid
        )
        # Set up table textures for training/testing.
        self._num_table_textures_train = num_table_textures_train
        self._num_table_textures_test = num_table_textures_test
        self.table_texture_ids = dict(
            train=[],
            test=[]
        )
        self.setup_table_textures()

        p.setAdditionalSearchPath(asset.getDataPath(), physicsClientId=self.cid)
        self.target_init_z = 0.05 + z_shift
        self.target_des_z = 0.3 + z_shift
        self.panda_init_z = 0.2

        self.fixed_target_init_pos = [0., 0., self.target_init_z]
        self.fixed_panda_init_pos = [0., -0.1, self.panda_init_z]

        self.target_id = p.loadURDF(
            fileName='cube.urdf',
            basePosition=self.fixed_target_init_pos,
            globalScaling=0.05,
            physicsClientId=self.cid
        )

        self.target_texture_id = p.loadTexture('./cube_grasping/asset/textures/grunge.png', physicsClientId=self.cid)
        print(f'target texture id: {self.target_texture_id}')
        p.changeVisualShape(self.target_id, -1, rgbaColor=[170/255, 100/255, 30/255, 1], textureUniqueId=self.target_texture_id,
                            physicsClientId=self.cid)

        # Set up distractors.
        self.setup_distractors(distractor_color)

        p.setGravity(0, 0, -9.8, physicsClientId=self.cid)
        p.setTimeStep(self.time_step, physicsClientId=self.cid)
        if self._image_obs:
            if self._view == 1:
                env_obs_spaces = {
                    'im_rgb_1': spaces.Box(0, 255, [3] + self._render_size, np.uint8),
                }
            if self._view == 3:
                env_obs_spaces = {
                    'im_rgb_3': spaces.Box(0, 255, [3] + self._render_size, np.uint8),
                }
            elif self._view == 'both':
                env_obs_spaces = {
                    'im_rgb_1': spaces.Box(0, 255, [3] + self._render_size, np.uint8),
                    'im_rgb_3': spaces.Box(0, 255, [3] + self._render_size, np.uint8),
                }
        else:
            env_obs_spaces = {
                'target_pos': spaces.Box(-np.inf, np.inf, (3,), np.float32)
            }
        self.observation_space = spaces.Dict({**env_obs_spaces, **self.panda.observation_spaces})
        self.action_space = self.panda.action_space

    def step(self, a):
        assert a.shape == (4,)
        a_pos, a_grip = a[:3], a[3]
        delta_pos = self.a_pos_scale * a_pos
        delta_grip = self.a_grip_scale * a_grip
        for _ in range(self.action_repeat):
            self.panda.step(delta_pos / self.action_repeat, delta_grip / self.action_repeat)
            p.stepSimulation(physicsClientId=self.cid)

        if self._real_time:
            time.sleep(self.time_step * self.action_repeat)

        obs = self.get_observation()
        info = self.get_info()

        # Reward function.
        reward = 0
        reward -= info['ee_target_dist'] # reward for getting closer to target obj
        reward += info['holding'] * 10 # reward for gripping the target obj
        if info['holding']:
            delta_target_z_scaled = info['target_z_scaled'] - self._old_target_z_scaled
            self._old_target_z_scaled = info['target_z_scaled']
            reward += delta_target_z_scaled * 100 # reward for lifting target obj up
        if info['success']:
            reward += 1000 # reward for reaching goal state
        # Add a time penalty to encourage faster termination of the episode.
        reward -= 0.5

        terminated = info['success']
        truncated = False
        return obs, reward, terminated, truncated, info

    def reset(self):
        # reset the manipulator
        if self._random_panda_init_pos:
            panda_init_pos = self.sample_panda_init_pos()
        else:
            panda_init_pos = self.fixed_panda_init_pos
        self.panda.reset(panda_init_pos)

        # reset the target object's position on the table.
        if self._random_target_init_pos:
            target_init_pos = self.sample_target_init_pos()
        else:
            target_init_pos = self.fixed_target_init_pos
        p.resetBasePositionAndOrientation(
            self.target_id,
            target_init_pos,
            p.getQuaternionFromEuler([0., 0., 0.], physicsClientId=self.cid),
            physicsClientId=self.cid
        )
        self._old_target_z_scaled = 0. # used in the reward function

        # Reset the distractors' positions on the table (if they exist).
        if self._add_distractors:
            while True:
                for distractor in self.distractors:
                    p.resetBasePositionAndOrientation(
                        distractor,
                        self.sample_distractor_init_pos(),
                        p.getQuaternionFromEuler([0., 0., 0.], physicsClientId=self.cid),
                        physicsClientId=self.cid
                    )
                p.performCollisionDetection(physicsClientId=self.cid)
                valid = True
                for i in self.distractors + [self.target_id]:
                    for j in self.distractors + [self.target_id]:
                        if i == j:
                            continue
                        valid = valid and len(
                            p.getContactPoints(bodyA=i, bodyB=j, physicsClientId=self.cid)
                        ) == 0
                if valid:
                    break
                else:
                    print(f'collision detected, re-sampling objects')


        # Randomly reset the table texture.
        if self._random_table_texture:
            random_texture = self.sample_table_texture()
            p.changeVisualShape(self.table_id, -1, textureUniqueId=random_texture, physicsClientId=self.cid)

        return self.get_observation(), self.get_info()

    def get_observation(self):
        panda_obs = self.panda.get_observation()
        # im_rgb: 0-255 (int), im_d: (float)
        if self._image_obs:
            env_obs = self.render()
        else:
            target_pos, _ = p.getBasePositionAndOrientation(self.target_id, physicsClientId=self.cid)
            env_obs = dict(
                target_pos=target_pos
            )
        return {**env_obs, **panda_obs}

    def get_info(self):
        # ee_pos: end effector position
        # target_pos: target object position
        # ee_grip:
        # ee_target_dist: distance between end effector and target object
        # holding: True/False, whether the end effector is holding the target object
        # target_z_scaled: how far up the target object is to goal z position,
        #   scaled s.t. 0 means no progress, >= 1 means we hit goal height
        ee_pos, _, ee_grip = self.panda.get_ee_pos_ori_grip()
        target_pos, _ = p.getBasePositionAndOrientation(self.target_id, physicsClientId=self.cid)
        ee_target_dist = np.linalg.norm(ee_pos - target_pos)
        holding = self.panda.get_is_holding(self.target_id)
        target_z_scaled = (target_pos[2] - self.target_init_z) / (self.target_des_z - self.target_init_z)
        return dict(
            target_pos=target_pos,
            ee_pos=ee_pos,
            ee_grip=ee_grip,
            ee_target_dist=ee_target_dist,
            holding=holding,
            target_z_scaled=target_z_scaled,
            success=holding and target_z_scaled >= 1.0
        )

    def render(self):
        far = 1000.
        near = 0.01
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=1.0,
            nearVal=near,
            farVal=far,
            physicsClientId=self.cid
        )
        cam_img_args = dict(
            width=self._render_size[0],
            height=self._render_size[1],
            projectionMatrix=projection_matrix,
            shadow=1,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=self.cid
        )
        if self._view == 1 or self._view == 'both':
            view_matrix1 = self.panda.get_ego_view_matrix()
            cam_img_args['viewMatrix'] = view_matrix1
            w, h, im_rgba, im_d, im_seg = p.getCameraImage(**cam_img_args)
            im_rgb_1 = np.transpose(im_rgba[:, :, :3], (2, 0, 1))
            if self._view == 1:
                return dict(
                    im_rgb_1=im_rgb_1,
                )
        if self._view == 3 or self._view == 'both':
            view_matrix3 = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.1, -0.3, 0.23],
                distance=0.41,
                yaw=35,
                pitch=-32,
                roll=0,
                upAxisIndex=2,
                physicsClientId=self.cid
            )
            cam_img_args['viewMatrix'] = view_matrix3
            w, h, im_rgba, im_d, im_seg = p.getCameraImage(**cam_img_args)
            im_rgb_3 = np.transpose(im_rgba[:, :, :3], (2, 0, 1))
            if self._view == 3:
                return dict(
                    im_rgb_3=im_rgb_3,
                )
        if self._view == 'both':
            return dict(
                im_rgb_1=im_rgb_1,
                im_rgb_3=im_rgb_3,
            )

    def sample_panda_init_pos(self):
        low = np.array([-0.1, -0.2, self.panda_init_z - 0.05])
        high = np.array([0.1, 0., self.panda_init_z + 0.05])
        return np.random.uniform(low, high)

    def sample_target_init_pos(self):
        low = np.array([-0.1, -0.4, self.target_init_z])
        high = np.array([0.1, -0.2, self.target_init_z])
        return np.random.uniform(low, high)

    def sample_distractor_init_pos(self):
        low = np.array([-0.25, -0.55, self.target_init_z])
        high = np.array([0.25, -0.05, self.target_init_z])
        return np.random.uniform(low, high)

    def setup_table_textures(self):
        texture_paths = glob.glob(os.path.join('./dtd', '**', '*.jpg'), recursive=True)
        # Shuffle all textures, and use the first half for training and the second
        # half for testing.
        random.seed(1)  # do NOT remove/change this
        random.shuffle(texture_paths)
        sampled_texture_paths = dict(
            train=texture_paths[:self._num_table_textures_train],
            test=texture_paths[100:100+self._num_table_textures_test] # skip the first 100
        )
        for split in 'train', 'test':
            for path in sampled_texture_paths[split]:
                self.table_texture_ids[split].append(p.loadTexture(path, physicsClientId=self.cid))

    def sample_table_color(self):
        """Return a randomly chosen table RGBA color."""
        train_rgb_colors = [
            (0, 0, 0),  # black
            (255, 0, 0),  # red
            (0, 0, 255),  # blue
            (0, 255, 255),  # cyan
            (128, 128, 128),  # gray
            (128, 128, 0),  # olive
            (128, 0, 128),  # purple
        ]
        test_rgb_colors = [
            (255, 255, 255),  # white
            (0, 255, 0),  # green
            (255, 255, 0),  # yellow
            (255, 0, 255),  # pink
            (128, 0, 0),  # maroon
            (0, 128, 0),  # dark green
            (0, 128, 128),  # teal
        ]
        if self._train:
            random_rgb = list(random.sample(train_rgb_colors, 1)[0])
        else:
            random_rgb = list(random.sample(test_rgb_colors, 1)[0])
        random_rgba = random_rgb.append(1)  # append alpha value to the color
        return random_rgba

    def sample_table_texture(self):
        """
        Return a randomly chosen texture ID from the Describable Textures Dataset (DTD)
        (source: https://www.robots.ox.ac.uk/~vgg/data/dtd/).
        """
        if self._train:
            texture_id = np.random.choice(self.table_texture_ids['train'])
        else:
            texture_id = np.random.choice(self.table_texture_ids['test'])
        return texture_id

    def color_to_rgba(self, color: str):
        """Returns the RGB+A (as a list) of a given color, e.g. 'red' -> [255/255, 0/255, 0/255, 1]."""
        if color == 'red':
            return [1, 0, 0, 1]
        elif color == 'green':
            return [0, 1, 0, 1]
        elif color == 'blue':
            return [0, 0, 1, 1]
        elif color == 'brown':
            return [170/255, 100/255, 30/255, 1]
        elif color == 'black':
            return [0, 0, 0, 1]
        elif color == 'white':
            return [1, 1, 1, 1]
        else:
            print("Bad color given.")
            exit(1)

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)

    def setup_distractors(self, distractor_color):
        """Instantiates the distractor objects, if applicable, with the correct color(s)."""
        if self._add_distractors:
            self.distractors = []
            if distractor_color == 'mix':  # 1 red, 1 green, 1 blue
                # Red distractor
                distractor = p.loadURDF(
                    fileName='cube-copy.urdf',
                    basePosition=self.sample_distractor_init_pos(), # random position
                    globalScaling=0.05,
                    physicsClientId=self.cid
                )
                p.changeVisualShape(distractor, -1, rgbaColor=self.color_to_rgba('red'), physicsClientId=self.cid)
                self.distractors.append(distractor)

                # Green distractor
                distractor = p.loadURDF(
                    fileName='cube-copy.urdf',
                    basePosition=self.sample_distractor_init_pos(), # random position
                    globalScaling=0.05,
                    physicsClientId=self.cid
                )
                p.changeVisualShape(distractor, -1, rgbaColor=self.color_to_rgba('green'), physicsClientId=self.cid)
                self.distractors.append(distractor)

                # Blue distractor
                distractor = p.loadURDF(
                    fileName='cube-copy.urdf',
                    basePosition=self.sample_distractor_init_pos(), # random position
                    globalScaling=0.05,
                    physicsClientId=self.cid
                )
                p.changeVisualShape(distractor, -1, rgbaColor=self.color_to_rgba('blue'), physicsClientId=self.cid)
                self.distractors.append(distractor)

            else: # all distractors same color
                for _ in range(3):
                    distractor = p.loadURDF(
                        fileName='cube-copy.urdf',
                        basePosition=self.sample_distractor_init_pos(), # random position
                        globalScaling=0.05,
                        physicsClientId=self.cid
                    )
                    p.changeVisualShape(distractor, -1, rgbaColor=self.color_to_rgba(distractor_color), physicsClientId=self.cid)
                    self.distractors.append(distractor)
