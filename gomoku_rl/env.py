from typing import Callable
from tensordict import TensorDict
import torch

from torchrl.data.tensor_specs import (
    CompositeSpec,
    DiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from .core import Gomoku


# 定义环境，并声明了环境的 observation_spec、action_spec 和 reward_spec，以及 reset 和 step 方法。
class GomokuEnv:
    def __init__(
        self,
        num_envs: int,
        board_size: int,
        device=None,
        action_pruning=None,
        use_temporal_feature: bool = False,
        temporal_num_steps: int = 3,
    ):
        self.gomoku = Gomoku(
            num_envs=num_envs,
            board_size=board_size,
            device=device,
            action_pruning=action_pruning,
        )
        self.use_temporal_feature = bool(use_temporal_feature)
        self.temporal_num_steps = int(temporal_num_steps)
        assert self.temporal_num_steps >= 1, "temporal_num_steps must be >= 1"

        self.observation_channels = (
            2 * self.temporal_num_steps if self.use_temporal_feature else 3
        )

        if self.use_temporal_feature:
            self.board_history = torch.zeros(
                num_envs,
                self.temporal_num_steps,
                board_size,
                board_size,
                device=self.device,
                dtype=torch.long,
            )
        else:
            self.board_history = None

        self.observation_spec = CompositeSpec(
            {
                "observation": UnboundedContinuousTensorSpec(
                    device=self.device,
                    shape=[num_envs, self.observation_channels, board_size, board_size],
                ),
                "action_mask": BinaryDiscreteTensorSpec(
                    n=board_size * board_size,
                    device=self.device,
                    shape=[num_envs, board_size * board_size],
                    dtype=torch.bool,
                ),
            },
            shape=[num_envs],
            device=self.device,
        )

        self.action_spec = DiscreteTensorSpec(
            board_size * board_size,
            shape=[num_envs],
            device=self.device,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=[num_envs, 1],
            device=self.device,
        )
        self._post_step = None

    @property
    def batch_size(self):
        return torch.Size((self.num_envs,))

    @property
    def board_size(self):
        return self.gomoku.board_size

    @property
    def device(self):
        return self.gomoku.device

    @property
    def num_envs(self):
        return self.gomoku.num_envs

    def _build_temporal_observation(self) -> torch.Tensor:
        # 历史第 0 维始终是当前局面，第 1 维是前一局面，以此类推。
        # 每个局面拆成两个通道：[black_plane, white_plane]
        black_planes = (self.board_history == 1).float()
        white_planes = (self.board_history == -1).float()
        stacked = torch.stack([black_planes, white_planes], dim=2)
        return stacked.flatten(start_dim=1, end_dim=2)

    def _build_observation(self) -> torch.Tensor:
        if not self.use_temporal_feature:
            return self.gomoku.get_encoded_board()
        return self._build_temporal_observation()

    def _init_or_clear_history(self, env_indices: torch.Tensor | None = None) -> None:
        if not self.use_temporal_feature:
            return

        if env_indices is None:
            self.board_history.zero_()
            self.board_history[:, 0] = self.gomoku.board
            return

        if env_indices.numel() == 0:
            return

        self.board_history[env_indices] = 0
        self.board_history[env_indices, 0] = self.gomoku.board[env_indices]

    def _push_current_board_to_history(self, update_mask: torch.Tensor) -> None:
        if not self.use_temporal_feature or not update_mask.any():
            return

        self.board_history[update_mask, 1:] = self.board_history[update_mask, :-1].clone()
        self.board_history[update_mask, 0] = self.gomoku.board[update_mask]

    def reset(self, env_indices: torch.Tensor | None = None) -> TensorDict:
        """Resets the specified game environments to their initial states, or all environments if none are specified.

        Args:
            env_indices (torch.Tensor | None, optional): Indices of environments to reset. Resets all if None. Defaults to None.

        Returns:
            TensorDict: A tensor dictionary containing the initial observations and action masks for all environments.
        """
        self.gomoku.reset(env_indices=env_indices)
        self._init_or_clear_history(env_indices=env_indices)

        tensordict = TensorDict(
            {
                "observation": self._build_observation(),
                "action_mask": self.gomoku.get_action_mask(),
            },
            self.batch_size,
            device=self.device,
        )
        return tensordict

    def step(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        """Advances the state of the environments by one timestep based on the actions provided in the `tensordict`.

        Args:
            tensordict (TensorDict): A dictionary containing tensors with the actions to be taken in each environment. May also include optional environment masks to specify which environments should be updated.

        Returns:
            TensorDict: output tensor dictionary containing the updated observations, action masks, and other information for all environments.
        """
        action: torch.Tensor = tensordict.get("action")
        env_mask: torch.Tensor = tensordict.get("env_mask", None)
        update_mask = (
            torch.ones_like(action, dtype=torch.bool) if env_mask is None else env_mask
        )

        episode_len = self.gomoku.move_count + 1  # (E,)
        win, illegal = self.gomoku.step(action=action, env_mask=env_mask)

        assert not illegal.any()

        self._push_current_board_to_history(update_mask=update_mask)

        done = win
        black_win = win & (episode_len % 2 == 1)
        white_win = win & (episode_len % 2 == 0)
        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(
            {
                "observation": self._build_observation(),
                "action_mask": self.gomoku.get_action_mask(),
                "done": done,
                "win": win,
                # reward is calculated later
                "stats": {
                    "episode_len": episode_len,
                    "black_win": black_win,
                    "white_win": white_win,
                },
            }
        )
        if self._post_step:
            self._post_step(tensordict)
        return tensordict

    def step_and_maybe_reset(
        self,
        tensordict: TensorDict,
        env_mask: torch.Tensor | None = None,
    ) -> TensorDict:
        """Simulates a single step of the game environment and resets the environment if the game ends.

        Args:
            tensordict (TensorDict): A dictionary containing tensors with the current observations, action masks, and actions for each environment.
            env_mask (torch.Tensor | None, optional): A 1D tensor specifying which environments should be updated. If `None`, all environments are updated.

        Returns:
            TensorDict: A dictionary containing tensors with the updated observations, action masks, and other relevant information for each environment.
            For environments that have concluded their game and are reset, the 'observation' key will reflect the new initial state,
            but **the 'done' flag remains set to True** to indicate the end of the previous game within this timestep.
        """

        if env_mask is not None:
            tensordict.set("env_mask", env_mask)
        next_tensordict = self.step(tensordict=tensordict)
        tensordict.exclude("env_mask", inplace=True)

        done: torch.Tensor = next_tensordict.get("done")  # (E,)
        env_ids = done.nonzero(as_tuple=False).flatten()
        reset_td = self.reset(env_indices=env_ids)
        next_tensordict.update(reset_td)  # no impact on training
        return next_tensordict

    def set_post_step(self, post_step: Callable[[TensorDict], None] | None = None):
        """Sets a function to be called after each step in the environment.

        Args:
            post_step (Callable[[TensorDict], None] | None, optional): A function that takes a tensor dictionary as input and performs some action. Defaults to None.
        """
        self._post_step = post_step
