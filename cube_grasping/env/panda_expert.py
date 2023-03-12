"""Generates simulated expert demonstrations in which a Franka Emika Panda robot arm grasps a cube on a tabletop."""

import cube_grasping.env
from cube_grasping.policy import CubeGraspingRobotExtractor
import gym
import ipdb
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
import os
import pickle
from tqdm import tqdm
import imageio
import argparse
import os


class PandaExpertPolicy:
    def compute_action(self, info, verbose=False):
        state = -1
        if info['holding']:
            state = 0
            a_grip = -0.25 * np.array([1.])
            a_pos = np.array([0., 0., 1.])
        else:
            ee_pos = info['ee_pos']
            target_pos = info['target_pos']
            if np.linalg.norm(ee_pos[:2] - target_pos[:2]) < 0.015 and ee_pos[2] < target_pos[2] + 0.01:
                state = 1
                a_grip = np.array([-1.])
                a_pos = np.zeros((3,))
            else:
                if info['ee_grip'] < 0.075:
                    a_grip = np.array([1.])
                else:
                    a_grip = np.array([0.])

                state = 3
                r = target_pos - np.array([0., 0., 0.01]) - ee_pos
                a_pos = r / np.linalg.norm(r)
        if verbose:
            print(f'state={state}')
        a = np.concatenate([a_pos, a_grip])
        return a


class PandaExpertCollector:
    def __init__(self, env, path, num_episodes, train_env_kwargs):
        self.env = env
        self.policy = PandaExpertPolicy()
        self.path = path
        self.train_env_kwargs = train_env_kwargs
        self._num_episodes = num_episodes
        self._i_max = self.env._max_episode_steps * self._num_episodes
        self._i = 0
        self.obs = {
            k: np.zeros((self._i_max,) + self.env.observation_space[k].shape, dtype=self.env.observation_space[k].dtype)
            for k in self.env.observation_space.spaces.keys() if k not in ['im_rgb_1', 'im_rgb_3']
        }
        if self.train_env_kwargs['image_obs']:
            for view in [1, 3]:
                self.obs[f'im_rgb_{view}'] = np.zeros((self._i_max,) + self.env.observation_space[f'im_rgb_{view}'].shape, dtype=self.env.observation_space[f'im_rgb_{view}'].dtype)
        self.acts = np.zeros((self._i_max,) + self.env.action_space.shape, dtype=self.env.action_space.dtype)

    def collect_data(self, debug=False):
        success = []
        lengths = []
        pbar = tqdm(range(self._num_episodes))
        for i_episode in pbar:
            _ = self.env.reset()
            t = 0
            terminated = False
            truncated = False
            while not terminated and not truncated:
                obs = self.env.get_observation()
                info = self.env.get_info()
                a = self.policy.compute_action(info, verbose=False)
                for k in obs.keys():
                    if k != 'im_rgb':
                        self.obs[k][self._i] = obs[k]
                if not debug:
                    if self.train_env_kwargs['image_obs']:
                        render_obs = self.env.env.render()
                        for view in [1, 3]:
                            self.obs[f'im_rgb_{view}'][self._i] = render_obs[f'im_rgb_{view}']
                self.acts[self._i] = a
                self._i += 1
                noise = np.random.normal(
                    loc=np.zeros_like(a),
                    scale=0.1
                )
                a_noise = a + noise
                _, _, terminated, truncated, info = self.env.step(a_noise)
                t += 1
            print("Num time steps to complete episode:", t)
            lengths.append(t)
            success.append(info['success'])
            pbar.set_description_str(f'success: {np.mean(success)} length: {np.mean(lengths):.3f}')

    def write(self):
        for k in self.obs.keys():
            self.obs[k] = self.obs[k][:self._i]
        self.acts = self.acts[:self._i]
        dir = os.path.dirname(self.path)
        if dir != '':
            os.makedirs(dir, exist_ok=True)
        print(f'writing {len(self.acts)} transitions to {self.path}')
        pickle.dump(
            dict(obs=self.obs, acts=self.acts, train_env_kwargs=self.train_env_kwargs),
            open(self.path, 'wb'),
            protocol=4
        )


def collect(args):
    train_env_kwargs = dict(
        gui=False,
        train=True,
        random_table_texture=args.random_table_texture,
        num_table_textures_train=5,
        random_target_init_pos=True,
        random_panda_init_pos=True,
        image_obs=True,
        add_distractors=args.add_distractors,
        z_shift=0.,
        distractor_color='mix',
    )
    env = gym.make('PandaEnv-v0', real_time=False, view='both', **train_env_kwargs)
    expert = PandaExpertCollector(env, args.data_path, args.num_episodes, train_env_kwargs)
    expert.collect_data(debug=args.debug)
    expert.write()


def visualize(args):
    data = pickle.load(open(args.data_path, 'rb'))
    for view in [1, 3]:
        imageio.mimwrite(
            uri=f'{os.path.splitext(args.data_path)[0]}_view{view}.mp4',
            ims=data['obs'][f'im_rgb_{view}'].transpose([0, 2, 3, 1])
        )


def main(args):
    collect(args)
    visualize(args)


def str_to_bool(s: str) -> bool:
    if s not in {'True', 'False'}:
        raise ValueError('Invalid boolean string argument given.')
    return s == 'True'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/panda_expert_demos.pkl')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--add_distractors', type=str_to_bool, default=False)
    parser.add_argument('--random_table_texture', type=str_to_bool, default=False)
    args = parser.parse_args()
    main(args)
