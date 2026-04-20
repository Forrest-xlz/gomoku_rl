from typing import Generator, Iterable, Union

from torch.optim import Optimizer, Adam, AdamW
import torch
from torch.nn import Parameter
from torchrl.modules import ProbabilisticActor, ValueOperator
from torch.distributions.categorical import Categorical
from torchrl.data import TensorSpec
from torchrl.modules import (
    EGreedyModule,
    QValueActor,
    ActorValueOperator,
    SafeModule,
)
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict import TensorDict
from omegaconf import DictConfig

from gomoku_rl.utils.module import (
    ValueNet,
    ActorNet,
    ResidualTower,
    PolicyHead,
    ValueHead,
    MyDuelingCnnDQNet,
)
from gomoku_rl.utils.misc import get_kwargs

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


def make_dataset_naive(tensordict: TensorDict, batch_size: int) -> Generator[TensorDict, None, None]:
    tensordict = tensordict.reshape(-1)
    assert tensordict.shape[0] >= batch_size
    perm = torch.randperm(
        (tensordict.shape[0] // batch_size) * batch_size,
        device=tensordict.device,
    ).reshape(-1, batch_size)
    for indices in perm:
        yield tensordict[indices]


def get_optimizer(cfg: DictConfig, params: Iterable[Parameter]) -> Optimizer:
    dict_cls: dict[str, Optimizer] = {
        "adam": Adam,
        "adamw": AdamW,
    }
    name: str = cfg.name.lower()
    assert name in dict_cls
    return dict_cls[name](params=params, **cfg.kwargs)
