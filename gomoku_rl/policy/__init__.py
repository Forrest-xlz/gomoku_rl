from .base import Policy
from .ppo import PPO
from .ppo_review import PPOReview
from .dqn import DQN

from torchrl.data.tensor_specs import DiscreteTensorSpec, TensorSpec
from omegaconf import DictConfig
import torch


def get_policy(
    name: str,
    cfg: DictConfig,
    action_spec: DiscreteTensorSpec,
    observation_spec: TensorSpec,
    device="cuda",
) -> Policy:
    """
    Retrieves a policy object based on the specified policy name, configuration,
    action and observation specifications, and device.
    """
    cls = Policy.REGISTRY[name.lower()]
    return cls(
        cfg=cfg,
        action_spec=action_spec,
        observation_spec=observation_spec,
        device=device,
    )


def get_pretrained_policy(
    name: str,
    cfg: DictConfig,
    action_spec: DiscreteTensorSpec,
    observation_spec: TensorSpec,
    checkpoint_path: str,
    device="cuda",
) -> Policy:
    """
    Initializes and returns a pretrained policy object and loads the checkpoint.
    """
    policy = get_policy(
        name=name,
        cfg=cfg,
        action_spec=action_spec,
        observation_spec=observation_spec,
        device=device,
    )
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return policy
