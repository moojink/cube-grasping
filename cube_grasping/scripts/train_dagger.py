"""
Trains a behavioral cloning policy with DAgger on the robot grasping task.

Usage:
    Train with default args (see cube_grasping/scripts/dagger_args.py for the default values):
        python cube_grasping/scripts/train_dagger.py --data_path=...
    Train with just hand-centric POV:
        python cube_grasping/scripts/train_dagger.py --data_path=... --use_view_1=True --use_view_3=False
    Train with just third-person POV:
        python cube_grasping/scripts/train_dagger.py --data_path=... --use_view_1=False --use_view_3=True
    Resume training:
        python cube_grasping/scripts/train_dagger.py --data_path=... --checkpoint_epoch=... --checkpoint_dir=...
        E.g., resuming the third experiment (zero-indexed) from epoch 250:
            python cube_grasping/scripts/train_dagger.py --data_path=./data/panda_expert_demos.pkl --checkpoint_epoch=250 --checkpoint_dir=logs/bc/2/
            (Note that this resumes training for both hand-centric and third-person POVs unless you
             specify otherwise by using the --use_view_1 and --use_view_3 flags.)
    Train with a specific log directory:
        python cube_grasping/scripts/train_dagger.py --data_path=... --log_dir=...
        E.g., writing logs to a specific directory:
            python cube_grasping/scripts/train_dagger.py --data_path=./data/panda_expert_demos.pkl --log_dir=logs/bc/
"""

import copy
import csv
import dagger_args
import cube_grasping.env
import gym
import imageio
import numpy as np
import os
import pickle
import torch
from copy import deepcopy
from cube_grasping.env.panda_expert import PandaExpertPolicy
from cube_grasping.policy import CubeGraspingRobotExtractor
from gym.wrappers import TransformObservation
from imitation.algorithms import bc_dict_dagger, dagger_dict
from imitation.data import rollout
from imitation.util import util
from stable_baselines3.common import policies
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def panda_normalize_obs(obs):
    assert len(obs.keys()) == 4
    assert 'im_rgb_1' in obs.keys() or 'im_rgb_3' in obs.keys()
    assert 'ee_pos_rel_base' in obs.keys()
    assert 'ee_grip' in obs.keys()
    assert 'contact_flags' in obs.keys()
    obs['ee_grip'] = (obs['ee_grip'] - 0.04) / 0.04
    return obs

class CubeGraspingExpertDataset(Dataset):
    def __init__(self, data, camera_view: int):
        self.obs = data['obs']
        self.acts = data['acts']
        # Select camera view: hand-centric (1) or third-person POV (3).
        if camera_view == 1:
            self.obs.pop('im_rgb_3', None)
            self.obs['im_rgb_1'] = self.obs.pop('im_rgb_1')
        elif camera_view == 3:
            self.obs.pop('im_rgb_1', None)
            self.obs['im_rgb_3'] = self.obs.pop('im_rgb_3')
        else:
            raise ValueError('Invalid camera view given: {}'.format(camera_view))
        # Normalize the observations.
        self.obs = panda_normalize_obs(self.obs)

    def __len__(self):
        return len(self.acts)

    def __getitem__(self, index):
        return dict(
            obs={k: self.obs[k][index] for k in self.obs.keys()},
            acts=self.acts[index]
        )


def update_log_dir(log_dir: str):
    """
    Updates the log directory by appending a new experiment number.
    Example: 'logs/bc/' -> 'logs/bc/1' if 'logs/bc/0' exists and
             is the only experiment done thus far.

    Args:
        log_dir: The old log directory string.

    Returns:
        The new log directory string.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sub_dirs = [sub_dir for sub_dir in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, sub_dir))]
    if len(sub_dirs) == 0:
        return log_dir + '0/'
    sub_dirs_as_ints = [int(s) for s in sub_dirs]
    last_sub_dir = max(sub_dirs_as_ints)
    return log_dir + str(last_sub_dir + 1) + '/'


def record_gif(venv, bc_policy, gif_log_path):
    """Records and saves a gif of a behavior cloning policy acting on a vectorized environment."""
    obs, info = venv.reset()
    images = []
    num_episodes = 5
    for _ in range(num_episodes):
        for _ in range(venv.envs[0].spec.max_episode_steps):
            action, _ = bc_policy.predict(obs)
            obs, _, _, _, info = venv.step(action)
            img_dict = venv.render()
            assert len(img_dict) == 1 # should only render one camera view
            _, img = img_dict.popitem()
            images.append(img)
            if info[0]['success'] == True:
                break
    imageio.mimsave(gif_log_path, [np.transpose(img, (1,2,0)) for img in images], fps=10)


def main(
    data_path: str,
    camera_view: int,
    num_dagger_rounds: int,
    num_epochs_per_round: int,
    num_trajectories_per_round: int,
    log_dir: str,
    checkpoint_epoch: int,
    checkpoint_dir: str,
    seed: int,
    test_type: str,
):
    """Train the behavioral cloning policy with DAgger.

    Args:
        data_path: Path to the expert policy data used for training.
        camera_view: If 1, use the hand-centric POV camera. Else if 3, use third-person POV.
        num_dagger_rounds: Number of DAgger rounds to train for.
        num_epochs_per_round: Number of epochs to train for per DAgger round.
        num_trajectories_per_round: Number of trajectories to generate per DAgger round.
        log_dir: Log directoyr for TensorBoard stats and policy demo gifs.
        checkpoint_epoch: The epoch number at which to resume training. If 0, we
            start fresh.
        checkpoint_dir: The directory containing the saved checkpoint.
        seed: The random seed for the training environment.
        test_type: Test variant: 'table_height', 'distractors', or 'table_texture'.

    Returns:
        Nothing.
    """
    assert camera_view == 1 or camera_view == 3
    data = pickle.load(open(data_path, 'rb'))
    # Set up train and test environments.
    train_env_kwargs = data['train_env_kwargs']
    train_env_kwargs['view'] = camera_view
    test_env_kwargs = deepcopy(train_env_kwargs)
    test_env_kwargs['train'] = False
    train_env_kwargs['seed'] = seed
    if test_type == 'table_height':
        # Create the kwargs for the 5 table heights: -0.1, -0.05, 0, +0.05, +0.1.
        test_env_kwargs_list = []
        for z_shift in [-0.1, -0.05, 0., 0.05, 0.1]:
            test_env_kwargs_new = copy.deepcopy(test_env_kwargs)
            test_env_kwargs_new['z_shift'] = z_shift
            test_env_kwargs_list.append(test_env_kwargs_new)
    elif test_type == 'distractors':
        # Create the kwargs for different distractor colors: red, green, blue, brown, white, black
        test_env_kwargs_list = []
        for distractor_color in ['red', 'green', 'blue', 'brown', 'white', 'black']:
            test_env_kwargs_new = copy.deepcopy(test_env_kwargs)
            test_env_kwargs_new['distractor_color'] = distractor_color
            test_env_kwargs_list.append(test_env_kwargs_new)
    elif test_type == 'table_texture':
        # Create the kwargs for a test set of 20 held-out table textures.
        test_env_kwargs_list = []
        for num_table_textures_test in [20]:
            test_env_kwargs_new = copy.deepcopy(test_env_kwargs)
            test_env_kwargs_new['num_table_textures_test'] = num_table_textures_test
            test_env_kwargs_list.append(test_env_kwargs_new)
    else:
        return ValueError
    expert_dataset = CubeGraspingExpertDataset(data, camera_view)
    n_data = len(expert_dataset)
    n_train = int(0.8 * n_data)
    n_test = n_data - n_train
    [train_data, test_data] = random_split(expert_dataset, [n_train, n_test])
    batch_size = 256
    num_batches_per_epoch = int(n_train / batch_size)
    # Create a wrapper for normalizing the observations.
    post_wrappers = [
        lambda env, idx: TransformObservation(env, panda_normalize_obs)
    ]
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
    env_name = "PandaEnv-v0"
    venv = util.make_vec_env(env_name, n_envs=1, env_kwargs=train_env_kwargs, post_wrappers=post_wrappers)
    # Create a bc_trainer. If we load a BC policy training checkpoint, this object will replace dagger_trainer.bc_trainer.
    bc_trainer = bc_dict_dagger.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy_class=policies.ActorCriticPolicy,
        policy_kwargs=dict(
            net_arch=[64, 64],
            features_extractor_class=CubeGraspingRobotExtractor,
        ),
    )
    # Set up TensorBoard logging.
    if camera_view == 1:
        log_dir += 'view_1/'
    elif camera_view == 3:
        log_dir += 'view_3/'
    # Load a saved policy and optimizer state if resuming training from checkpoint.
    if checkpoint_epoch > 0:
        if camera_view == 1:
            checkpoint_dir += 'view_1/'
        elif camera_view == 3:
            checkpoint_dir += 'view_3/'
        saved_policy_state_path = checkpoint_dir + 'bc_policy_state.epoch_num=' + str(checkpoint_epoch)
        saved_policy_state = torch.load(saved_policy_state_path)
        bc_trainer.policy.load_state_dict(saved_policy_state)
        saved_optimizer_state_path = checkpoint_dir + 'bc_optimizer_state.epoch_num=' + str(checkpoint_epoch)
        saved_optimizer_state = torch.load(saved_optimizer_state_path)
        bc_trainer.optimizer.load_state_dict(saved_optimizer_state)
        # Update `log_dir` so that we save logs to the directory containing the checkpoint.
        log_dir = checkpoint_dir
        # Set model to training mode.
        bc_trainer.policy.train()
    # Note: Be careful to initialize this SummaryWriter only after `log_dir` has been fully updated.
    # In other words, make sure that `log_dir` is not reassigned after the SummaryWriter is initialized
    # because then we'd be writing logs to the wrong directory.
    tb_writer = SummaryWriter(log_dir=log_dir)
    print('\nLogging to directory {}\n'.format(log_dir))
    with open(log_dir + 'eval.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['epoch_num', 'mean_reward', 'success_rate'])
    for i, test_env_kwargs in enumerate(test_env_kwargs_list):
        with open(log_dir + f'test_{i}.csv', 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(['epoch_num', 'mean_reward', 'success_rate'])
    # Create environments for evaluation on the training distribution and test distributions.
    # We call the former "eval" and the latter "test".
    num_episodes_eval = 20
    venv_eval_train_dist = util.make_vec_env(env_name, n_envs=1, env_kwargs=train_env_kwargs, post_wrappers=post_wrappers)
    test_venvs_list = []
    for test_env_kwargs in test_env_kwargs_list:
        test_venv = util.make_vec_env(env_name, n_envs=1, env_kwargs=test_env_kwargs, post_wrappers=post_wrappers)
        test_venvs_list.append(test_venv)
    # Set up the DAgger trainer.
    dagger_trainer = dagger_dict.DAggerTrainer(
        env=TransformObservation(gym.make(env_name, **train_env_kwargs), panda_normalize_obs),
        scratch_dir=log_dir,
        beta_schedule=dagger_dict.LinearBetaSchedule(num_dagger_rounds),
        batch_size=batch_size,
        policy_class=policies.ActorCriticPolicy,
        policy_kwargs=dict(
            net_arch=[64, 64],
            features_extractor_class=CubeGraspingRobotExtractor,
        ),
        optimizer_kwargs=dict()
    )
    # Set the dagger_trainer's bc_trainer to the loaded one if applicable.
    if checkpoint_epoch > 0:
        dagger_trainer.bc_trainer = bc_trainer
    expert_policy = PandaExpertPolicy()
    # Define a callback function that we run after every X training epochs.
    # X=5 by default; to change this, modify variable `EVAL_INTERVAL` in cube-grasping/imitation/src/imitation/algorithms/bc_dict_dagger.py.
    def on_epoch_end_callback(epoch_num):
        # Set model to evaluation mode.
        dagger_trainer.bc_trainer.policy.eval()
        with torch.no_grad():
            # Get the loss over the test set.
            # Note: `bc_trainer._calculate_loss` returns the *average* loss over a batch.
            test_loss = 0.
            for batch in test_loader:
                batch_size = batch['acts'].shape[0]
                batch_avg_loss, _ = dagger_trainer.bc_trainer._calculate_loss(batch['obs'], batch['acts'])
                test_loss += batch_avg_loss * batch_size
            test_loss /= n_test
            tb_writer.add_scalar('loss (test)', test_loss, epoch_num)
            # Record a gif of the current policy run on the training and test distributions.
            gif_log_dir = log_dir + 'gif/'
            if not os.path.exists(gif_log_dir):
                os.makedirs(gif_log_dir)
            gif_log_path_train_dist = gif_log_dir + 'bc_policy.epoch_num=' + str(epoch_num) + '.eval.gif'
            record_gif(venv_eval_train_dist, dagger_trainer.bc_trainer.policy, gif_log_path_train_dist)
            for i, test_venv in enumerate(test_venvs_list):
                gif_log_path_test = gif_log_dir + 'bc_policy.epoch_num=' + str(epoch_num) + f'.test_{i}.gif'
                record_gif(test_venv, dagger_trainer.bc_trainer.policy, gif_log_path_test)
            # Log evaluation stats for the training distribution.
            eval_sample_until = rollout.min_episodes(num_episodes_eval)
            trajs_train_dist = rollout.generate_trajectories(dagger_trainer.bc_trainer.policy, venv_eval_train_dist, eval_sample_until)
            assert len(trajs_train_dist) == num_episodes_eval
            stats_train_dist = rollout.rollout_stats(trajs_train_dist)
            print("\n**** Eval Stats (train dist):\n", stats_train_dist)
            tb_writer.add_scalar('eval/reward', stats_train_dist['return_mean'], epoch_num)
            tb_writer.add_scalar('eval/success_rate', stats_train_dist['success_rate'], epoch_num)
            with open(log_dir + 'eval.csv', 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow([epoch_num, stats_train_dist['return_mean'], stats_train_dist['success_rate']])
            # Log evaluation stats for the test distribution.
            for i, test_venv in enumerate(test_venvs_list):
                test_trajs = rollout.generate_trajectories(dagger_trainer.bc_trainer.policy, test_venv, eval_sample_until)
                test_stats = rollout.rollout_stats(test_trajs)
                print(f"\n**** Test Stats, i={i}:\n", test_stats)
                tb_writer.add_scalar(f'test/reward_{i}', test_stats['return_mean'], epoch_num)
                tb_writer.add_scalar(f'test/success_rate_{i}', test_stats['success_rate'], epoch_num)
                with open(log_dir + f'test_{i}.csv', 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',')
                    csvwriter.writerow([epoch_num, test_stats['return_mean'], test_stats['success_rate']])
            # Save policy and optimizer state.
            dagger_trainer.bc_trainer.save_policy_state(log_dir + 'bc_policy_state.epoch_num=' + str(epoch_num))
        # Set model back to training mode.
        dagger_trainer.bc_trainer.policy.train()
    last_step = 0 # for logging purposes
    # Launch DAgger training.
    for i in tqdm(range(num_dagger_rounds)):
        print("Starting DAgger round {}...".format(i))
        collector = dagger_trainer.get_trajectory_collector()
        print("Generating more trajectories...")
        for _ in tqdm(range(num_trajectories_per_round)):
            obs, info = collector.reset()
            info = collector.env.get_info()
            done = False
            while not done:
                expert_action = expert_policy.compute_action(info, verbose=False)
                obs, _, terminated, truncated, info = collector.step(expert_action)
                done = terminated or truncated
        print("Training BC policy...")
        _, last_step = dagger_trainer.extend_and_update(
            log_dir=log_dir,
            n_epochs=num_epochs_per_round,
            on_epoch_end=on_epoch_end_callback,
            last_step=last_step,
            round_num=i,
        )


if __name__ == '__main__':
    # Parse command-line arguments.
    args = dagger_args.get_args()
    args.log_dir = update_log_dir(args.log_dir) # increment experiment number
    main_args_dict = {
        'data_path': args.data_path,
        'camera_view': 1,
        'num_epochs_per_round': args.num_epochs_per_round,
        'num_dagger_rounds': args.num_dagger_rounds,
        'num_trajectories_per_round': args.num_trajectories_per_round,
        'log_dir': args.log_dir,
        'checkpoint_epoch': args.checkpoint_epoch,
        'checkpoint_dir': args.checkpoint_dir,
        'seed': args.seed,
        'test_type': args.test_type,
    }
    # Use hand-centric POV camera if use_view_1=True. Use third-person POV if use_view_3=True.
    if args.use_view_1:
        main_args_dict['camera_view'] = 1
        main(**main_args_dict)
    if args.use_view_3:
        main_args_dict['camera_view'] = 3
        main(**main_args_dict)
