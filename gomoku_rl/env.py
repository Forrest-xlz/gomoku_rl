from typing import Callable

import torch
from tensordict import TensorDict
from torchrl.data.tensor_specs import (
    BinaryDiscreteTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)

from .core import Gomoku


class GomokuEnv:
    """Gomoku environment wrapper.

    Observation modes:
    - use_temporal_feature = False:
        keep the original legacy 3-channel observation returned by core:
        [current_player_board, opponent_board, last_move_one_hot]

    - use_temporal_feature = True:
        build a new observation:
        [current_player_board, opponent_board, last_1_move_one_hot, ..., last_n_move_one_hot]
        total channels = 2 + temporal_num_steps
    """

    def __init__(
        self,
        num_envs: int,
        board_size: int,
        device=None,
        action_pruning=None,
        use_temporal_feature: bool = False,
        temporal_num_steps: int = 6,
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
            2 + self.temporal_num_steps if self.use_temporal_feature else 3
        )

        if self.use_temporal_feature:
            self.move_history = torch.zeros(
                num_envs,
                self.temporal_num_steps,
                board_size,
                board_size,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self.move_history = None

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

    def _get_current_board_planes(self) -> torch.Tensor:
        # Core legacy encoding is [current_player, opponent, last_move].
        # We only keep the first two planes here.
        encoded = self.gomoku.get_encoded_board()
        return encoded[:, :2]

    def _build_observation(self) -> torch.Tensor:
        if not self.use_temporal_feature:
            return self.gomoku.get_encoded_board()
        current_board = self._get_current_board_planes()
        return torch.cat([current_board, self.move_history], dim=1)

    def _clear_history(self, env_indices: torch.Tensor | None = None) -> None:
        if not self.use_temporal_feature:
            return
        if env_indices is None:
            self.move_history.zero_()
            return
        if env_indices.numel() == 0:
            return
        self.move_history[env_indices] = 0.0

    def _push_last_move_to_history(self, update_mask: torch.Tensor) -> None:
        if not self.use_temporal_feature or not update_mask.any():
            return

        env_ids = update_mask.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        # Shift old history: slot 0 is most recent move.
        if self.temporal_num_steps > 1:
            self.move_history[env_ids, 1:] = self.move_history[env_ids, :-1].clone()
        self.move_history[env_ids, 0] = 0.0

        last_move = self.gomoku.last_move[env_ids]
        x = last_move // self.board_size
        y = last_move % self.board_size
        self.move_history[env_ids, 0, x, y] = 1.0

    def reset(self, env_indices: torch.Tensor | None = None) -> TensorDict:
        self.gomoku.reset(env_indices=env_indices)
        self._clear_history(env_indices=env_indices)
        tensordict = TensorDict(
            {
                "observation": self._build_observation(),
                "action_mask": self.gomoku.get_action_mask(),
            },
            self.batch_size,
            device=self.device,
        )
        return tensordict

    def step(self, tensordict: TensorDict) -> TensorDict:
        action: torch.Tensor = tensordict.get("action")
        env_mask: torch.Tensor = tensordict.get("env_mask", None)
        update_mask = (
            torch.ones_like(action, dtype=torch.bool) if env_mask is None else env_mask
        )

        episode_len = self.gomoku.move_count + 1
        win, illegal = self.gomoku.step(action=action, env_mask=env_mask)
        assert not illegal.any()

        self._push_last_move_to_history(update_mask=update_mask)

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
        if env_mask is not None:
            tensordict.set("env_mask", env_mask)
        next_tensordict = self.step(tensordict=tensordict)
        tensordict.exclude("env_mask", inplace=True)

        done: torch.Tensor = next_tensordict.get("done")
        env_ids = done.nonzero(as_tuple=False).flatten()
        reset_td = self.reset(env_indices=env_ids)
        next_tensordict.update(reset_td)
        return next_tensordict

    def set_post_step(self, post_step: Callable[[TensorDict], None] | None = None):
        self._post_step = post_step
