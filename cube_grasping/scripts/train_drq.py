import copy
import csv
import cube_grasping.env
import dmc2gym
import gym
import hydra
import math
import numpy as np
import os
import pickle as pkl
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.wrappers import TransformObservation

# Hack to import modules from drq submodule.
sys.path.append('./drq')
import drq as drq
import utils as utils
from logger import Logger
from replay_buffer import ReplayBuffer
from video import VideoRecorder

torch.backends.cudnn.benchmark = True


def panda_normalize_obs(obs):
    assert (len(obs.keys()) == 4 and 'im_rgb' in obs.keys()) or (len(obs.keys()) == 5 and 'im_rgb1' in obs.keys() and 'im_rgb3' in obs.keys())
    assert 'ee_pos_rel_base' in obs.keys()
    assert 'ee_grip' in obs.keys()
    assert 'contact_flags' in obs.keys()

    obs['ee_grip'] = (obs['ee_grip'] - 0.04) / 0.04
    return obs

def make_env(cfg):
    train_env_kwargs = cfg.train_env_kwargs # args for env setup
    env = TransformObservation(gym.make('PandaEnv-v0', **train_env_kwargs), panda_normalize_obs)
    env = utils.ActionRepeatWrapper(env, cfg.action_repeat, cfg.agent.params.discount)
    env = utils.FrameStack(cfg.view, env, k=cfg.frame_stack)

    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env

def make_test_env(test_env_kwargs, cfg):
    env = TransformObservation(gym.make('PandaEnv-v0', **test_env_kwargs), panda_normalize_obs)
    env = utils.ActionRepeatWrapper(env, cfg.action_repeat, cfg.agent.params.discount)
    env = utils.FrameStack(cfg.view, env, k=cfg.frame_stack)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    return env

def make_test_envs_list(cfg):
    train_env_kwargs = cfg.train_env_kwargs # args for env setup
    test_env_kwargs = dict.copy(dict(train_env_kwargs))
    test_env_kwargs['train'] = False
    test_envs_list = []
    if cfg.test_type == 'table_height':
        # Create the kwargs for the 5 table heights: -0.1, -0.05, 0, +0.05, +0.1
        for z_shift in [-0.1, -0.05, 0., 0.05, 0.1]:
            test_env_kwargs_new = copy.deepcopy(test_env_kwargs)
            test_env_kwargs_new['z_shift'] = z_shift
            test_envs_list.append(make_test_env(test_env_kwargs_new, cfg))
    elif cfg.test_type == 'table_texture':
        # Create the kwargs for an env with 20 test table textures
        for num_table_textures_test in [20]:
            test_env_kwargs_new = copy.deepcopy(test_env_kwargs)
            test_env_kwargs_new['num_table_textures_test'] = num_table_textures_test
            test_envs_list.append(make_test_env(test_env_kwargs_new, cfg))
    elif cfg.test_type == 'distractors':
        # Create the kwargs for different distractor colors: red, green, blue, brown, white, black, mix
        for distractor_color in ['red', 'green', 'blue', 'brown', 'white', 'black', 'mix']:
            test_env_kwargs_new = copy.deepcopy(test_env_kwargs)
            test_env_kwargs_new['distractor_color'] = distractor_color
            test_envs_list.append(make_test_env(test_env_kwargs_new, cfg))
    else:
        return ValueError
    return test_envs_list


def update_log_dir(log_dir: str):
    """
    Updates the log directory by appending a new experiment number.
    Example: 'logs/bc/' -> 'logs/bc/1' if 'logs/bc/0' exists and
             is the only experiment done thus far.
    Also adds a trailing '/' if it's missing.

    Args:
        log_dir: The old log directory string.

    Returns:
        The new log directory string.
    """

    # Add '/' to the end of log_dir if it's missing.
    if log_dir[-1] != '/':
        log_dir += '/'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sub_dirs = [sub_dir for sub_dir in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, sub_dir)) and sub_dir.isdigit()]
    if len(sub_dirs) == 0:
        return log_dir + '0/'
    sub_dirs_as_ints = [int(s) for s in sub_dirs]
    last_sub_dir = max(sub_dirs_as_ints)
    return log_dir + str(last_sub_dir + 1) + '/'


class Workspace(object):
    def __init__(self, cfg):
        log_dir = os.getcwd() # e.g., 'logs/drq/'
        self.log_dir = update_log_dir(log_dir) # increment experiment number

        os.makedirs(self.log_dir, exist_ok=True)

        print(f'workspace: {self.log_dir}')

        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg)
        self.test_envs_list = make_test_envs_list(cfg)
        for i in range(len(self.test_envs_list)):
            with open(self.log_dir + f'test_{i}.csv', 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(['frame', 'reward', 'success_rate'])

        self.logger = Logger(self.log_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name,
                             action_repeat=cfg.action_repeat)

        cfg.agent.params.obs_shape = self.env.observation_space.shape
        cfg.agent.params.action_shape = self.env.action_space.shape
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(cfg.view,
                                          self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          cfg.replay_buffer_capacity,
                                          self.cfg.image_pad, self.device)

        self.video_recorder = VideoRecorder(
            cfg.view,
            self.log_dir if cfg.save_video else None
        )
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        num_success = 0
        self.video_recorder.init(enabled=True)
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, info = self.env.step(action)
                if episode < 5: # only record 5 videos total
                    self.video_recorder.record(self.env)
                episode_reward += reward
                episode_step += 1

            if info['success'] == True:
                num_success += 1
            average_episode_reward += episode_reward
        self.video_recorder.save(f'{self.step}')
        average_episode_reward /= self.cfg.num_eval_episodes
        success_rate = num_success / self.cfg.num_eval_episodes
        self.logger.eval_log('eval/reward', average_episode_reward,
                        self.step, log_frequency=1)
        self.logger.eval_log('eval/success_rate', success_rate,
                        self.step, log_frequency=1)
        self.logger.dump(self.step)

    def test(self):
        for i, test_env in enumerate(self.test_envs_list):
            average_episode_reward = 0
            num_success = 0
            self.video_recorder.init(enabled=True)
            for episode in range(self.cfg.num_eval_episodes):
                obs = test_env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                while not done:
                    with utils.eval_mode(self.agent):
                        action = self.agent.act(obs, sample=False)
                    obs, reward, done, info = test_env.step(action)
                    if episode < 5: # only record 5 videos total
                        self.video_recorder.record(test_env)
                    episode_reward += reward
                    episode_step += 1

                if info['success'] == True:
                    num_success += 1
                average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}-test_{i}')
            average_episode_reward /= self.cfg.num_eval_episodes
            success_rate = num_success / self.cfg.num_eval_episodes
            self.logger.test_log(f'test/reward_{i}', average_episode_reward,
                            self.step, log_frequency=1)
            self.logger.test_log(f'test/success_rate_{i}', success_rate,
                            self.step, log_frequency=1)
            with open(self.log_dir + f'test_{i}.csv', 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow([self.step * self.cfg.action_repeat, average_episode_reward, success_rate])


    def save_checkpoint(self):
        self.agent.save_checkpoint(self.log_dir, self.step)

    def run(self):
        start_eval_logs = False # whether we should start logging eval metrics
        episode, episode_reward, episode_step, done = 0, 0, 1, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            # Evaluate the agent periodically and save checkpoint.
            if self.step % self.cfg.eval_frequency == 0 and start_eval_logs:
                self.logger.eval_log('eval/episode', episode, self.step)
                self.evaluate()
                self.test()
                self.save_checkpoint()
            # Check whether the training episode is complete.
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))
                    # We need `start_eval_logs` to start eval logs a bit later because
                    # if we start eval logs immediately upon starting training, then we
                    # get an error where the _dump_to_csv() function in the logger hasn't
                    # yet saved the CSV headers seen during training.
                    if self.step > self.cfg.num_seed_steps:
                        start_eval_logs = True

                if self.step > 0:
                    self.logger.log('train/reward', episode_reward,
                                    self.step)

                obs = self.env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                if self.step > 0:
                    self.logger.log('train/episode', episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger,
                                      self.step)

            next_obs, reward, done, info = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

@hydra.main(config_path='../../drq/ego_config.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
