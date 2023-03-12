import gym
import torch
from torch import nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac import SAC
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TensorDict, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from torchvision.models import resnet18
from torchinfo import summary
import ipdb


class ResNet18(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim=64):
        super(ResNet18, self).__init__(observation_space, features_dim=features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False)
        n_input_channels = observation_space.shape[0]

        encoder = resnet18(pretrained=False)
        self.cnn = nn.Sequential(
            *(list(encoder.children())[:-1]),   # remove classification layer
            nn.Flatten(),
            nn.Linear(512, features_dim, bias=True)
        )
        # print(summary(self.cnn, input_size=(256, 3, 128, 128)))
        self.features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.cnn(observations)


class CubeGraspingRobotExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(CubeGraspingRobotExtractor, self).__init__(observation_space, features_dim=1)  # dummy features_dim

        total_concat_size = 0
        extractors = {}
        for key, subspace in observation_space.spaces.items():
            if key in ['im_rgb_1', 'im_rgb_3']:
                extractors[key] = ResNet18(subspace)
                total_concat_size += extractors[key].features_dim
            elif key in ['ee_grip', 'ee_pos_rel_base', 'contact_flags']:
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)
            else:
                raise ValueError

        self.extractors = nn.ModuleDict(extractors)

        self.features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return torch.cat(encoded_tensor_list, dim=1)

