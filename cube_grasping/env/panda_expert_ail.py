import ego.env
from ego.policy import EgoRobotExtractor
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
        self.episodes = []

    def _fill_obs_with_views(self, obs):
        obs.pop('im_rgb')
        for view in [1, 3]:
            render_obs = self.env.env._render(view)
            obs[f'im_rgb_{view}'] = render_obs['im_rgb']

    def collect_data(self, debug=False):
        success = []
        lengths = []
        pbar = tqdm(range(self._num_episodes))
        for i_episode in pbar:
            episode = []
            obs = self.env.reset()
            self._fill_obs_with_views(obs)
            t = 0
            done = False
            while not done:
                info = self.env.get_info()
                a = self.policy.compute_action(info, verbose=False)
                noise = np.random.normal(
                    loc=np.zeros_like(a),
                    scale=0.1
                )
                a_noise = a + noise
                next_obs, _, done, info = self.env.step(a_noise)
                self._fill_obs_with_views(next_obs)
                episode.append((obs, a, next_obs))
                obs = next_obs
                t += 1
            print("Num time steps to complete episode:", t)
            self.episodes.append(episode)
            lengths.append(t)
            success.append(info['success'])
            pbar.set_description_str(f'success: {np.mean(success)} length: {np.mean(lengths):.3f}')

    def write(self):
        dir = os.path.dirname(self.path)
        if dir != '':
            os.makedirs(dir, exist_ok=True)
        print(f'writing {self._num_episodes} episodes to {self.path}')
        pickle.dump(
            dict(episodes=self.episodes, train_env_kwargs=self.train_env_kwargs),
            open(self.path, 'wb'),
            protocol=4
        )


def collect(args):
    train_env_kwargs = dict(
        gui=True,
        train=True,
        random_table_texture=False,
        num_table_textures_train=5,
        random_target_init_pos=True,
        random_panda_init_pos=True,
        image_obs=True,
        add_distractors=False,
        z_shift=0.,
        distractor_color='blue',
    )
    env = gym.make('PandaEnv-v0', real_time=False, view=1, **train_env_kwargs)
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
    # visualize(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/panda_expert_ail_v0.pkl')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args)
