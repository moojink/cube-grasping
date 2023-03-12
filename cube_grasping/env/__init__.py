import gym
from gym.envs.registration import register

register(
    id='PandaEnv-v0',
    entry_point='cube_grasping.env.panda_env:PandaEnv',
    max_episode_steps=40,
)
