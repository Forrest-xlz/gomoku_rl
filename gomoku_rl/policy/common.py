from typing import Generator, Iterable, Union

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.distributions.categorical import Categorical
from torch.nn import Parameter
from torch.optim import Adam, AdamW, Optimizer
from torchrl.data import TensorSpec
from torchrl.modules import (
    ActorValueOperator,
    EGreedyModule,
    ProbabilisticActor,
    QValueActor,
    SafeModule,
    ValueOperator,
)

from gomoku_rl.utils.misc import get_kwargs
from gomoku_rl.utils.module import (
    ActorNet,
    MyDuelingCnnDQNet,
    PolicyHead,
    ResidualTower,
    ValueHead,
    ValueNet,
)

DeviceLike = Union[torch.device, str, int, None]


def _get_in_channels(observation_spec: TensorSpec) -> int:
    return int(observation_spec["observation"].shape[-3])


def make_dqn_actor(
    cfg: DictConfig,
    action_spec: TensorSpec,
    observation_spec: TensorSpec,
    device: DeviceLike,
):
    net_kwargs = get_kwargs(cfg, "num_residual_blocks", "num_channels")

    net = MyDuelingCnnDQNet(
        in_channels=_get_in_channels(observation_spec),
        out_features=action_spec.space.n,
        **net_kwargs,
    )

    actor = QValueActor(
        net,
        spec=action_spec,
        action_mask_key="action_mask",
    ).to(device)

    return actor


def make_egreedy_actor(
    actor: TensorDictModule,
    action_spec: TensorSpec,
    eps_init: float = 1.0,
    eps_end: float = 0.10,
    annealing_num_steps: int = 1000,
):
    explorative_policy = TensorDictSequential(
        actor,
        EGreedyModule(
            spec=action_spec,
            eps_init=eps_init,
            eps_end=eps_end,
            annealing_num_steps=annealing_num_steps,
            action_mask_key="action_mask",
        ),
    )

    return explorative_policy


def make_ppo_actor(
    cfg: DictConfig,
    action_spec: TensorSpec,
    observation_spec: TensorSpec,
    device: DeviceLike,
):
    in_channels = _get_in_channels(observation_spec)

    actor_net = ActorNet(
        residual_tower=ResidualTower(
            in_channels=in_channels,
            num_channels=cfg.num_channels,
            num_residual_blocks=cfg.num_residual_blocks,
        ),
        out_features=action_spec.space.n,
        num_channels=cfg.num_channels,
    ).to(device)

    policy_module = TensorDictModule(
        module=actor_net,
        in_keys=["observation", "action_mask"],
        out_keys=["probs"],
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["probs"],
        distribution_class=Categorical,
        return_log_prob=True,
    )

    return policy_module


def make_critic(
    cfg: DictConfig,
    observation_spec: TensorSpec,
    device: DeviceLike,
):
    in_channels = _get_in_channels(observation_spec)

    value_net = ValueNet(
        residual_tower=ResidualTower(
            in_channels=in_channels,
            num_channels=cfg.num_channels,
            num_residual_blocks=cfg.num_residual_blocks,
        ),
        num_channels=cfg.num_channels,
    ).to(device)

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    return value_module


def make_ppo_ac(
    cfg: DictConfig,
    action_spec: TensorSpec,
    observation_spec: TensorSpec,
    device: DeviceLike,
):
    in_channels = _get_in_channels(observation_spec)

    residual_tower = ResidualTower(
        in_channels=in_channels,
        num_channels=cfg.num_channels,
        num_residual_blocks=cfg.num_residual_blocks,
    ).to(device)

    common_module = SafeModule(
        module=residual_tower,
        in_keys=["observation"],
        out_keys=["hidden"],
    )

    policy_head = PolicyHead(
        out_features=action_spec.space.n,
        num_channels=cfg.num_channels,
    ).to(device)

    policy_module = TensorDictModule(
        module=policy_head,
        in_keys=["hidden", "action_mask"],
        out_keys=["probs"],
    )

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=["probs"],
        distribution_class=Categorical,
        return_log_prob=True,
    )

    value_head = ValueHead(num_channels=cfg.num_channels).to(device)

    value_module = ValueOperator(
        module=value_head,
        in_keys=["hidden"],
    )

    return ActorValueOperator(common_module, policy_module, value_module)


def make_dataset_naive(
    tensordict: TensorDict,
    batch_size: int,
) -> Generator[TensorDict, None, None]:
    """
    Shuffle a TensorDict and yield PPO minibatches.

    Old behavior:
        Only full minibatches were used:
            floor(num_samples / batch_size) * batch_size
        Tail samples were dropped.

    New behavior:
        All samples are used.
        The last minibatch may be smaller than batch_size.

    Example:
        num_samples = 32256
        batch_size = 4096

        Old:
            7 minibatches, 3584 samples dropped.

        New:
            8 minibatches, last minibatch has 3584 samples.
    """
    tensordict = tensordict.reshape(-1)

    batch_size = int(batch_size)
    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"

    num_samples = int(tensordict.shape[0])
    assert num_samples > 0, "make_dataset_naive received an empty TensorDict."

    perm = torch.randperm(num_samples, device=tensordict.device)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        indices = perm[start:end]
        yield tensordict[indices]


def get_optimizer(cfg: DictConfig, params: Iterable[Parameter]) -> Optimizer:
    dict_cls: dict[str, Optimizer] = {
        "adam": Adam,
        "adamw": AdamW,
    }

    name: str = cfg.name.lower()
    assert name in dict_cls

    return dict_cls[name](params=params, **cfg.kwargs)
