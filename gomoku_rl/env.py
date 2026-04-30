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

LEGACY_MODE = "legacy"
TEMPORAL_MOVE_HISTORY_MODE = "temporal_move_history"
AZ_HISTORY_MODE = "az_history"


class GomokuEnv:
    """Gomoku environment wrapper.

    Supported observation modes
    ---------------------------
    1) legacy:
       [current_player_board, opponent_board, last_move_one_hot]

    2) temporal_move_history:
       [current_player_board, opponent_board,
        last_1_move_one_hot, ..., last_n_move_one_hot]
       total channels = 2 + n

    3) az_history:
       [black_board_t, black_board_t-1, ..., black_board_t-(n-1),
        white_board_t, white_board_t-1, ..., white_board_t-(n-1),
        side_to_move_plane]
       where side_to_move_plane = all ones if black to move, else all zeros.
       total channels = 2 * n + 1
    """

    def __init__(
        self,
        num_envs: int,
        board_size: int,
        device=None,
        action_pruning=None,
        use_temporal_feature: bool = False,
        temporal_num_steps: int = 6,
        observation_mode: str | None = None,
    ):
        self.gomoku = Gomoku(
            num_envs=num_envs,
            board_size=board_size,
            device=device,
            action_pruning=action_pruning,
        )

        self.observation_mode = self._resolve_observation_mode(
            observation_mode=observation_mode,
            use_temporal_feature=use_temporal_feature,
        )
        self.temporal_num_steps = int(temporal_num_steps)
        assert self.temporal_num_steps >= 1, "temporal_num_steps must be >= 1"

        if self.observation_mode == LEGACY_MODE:
            self.observation_channels = 3
        elif self.observation_mode == TEMPORAL_MOVE_HISTORY_MODE:
            self.observation_channels = 2 + self.temporal_num_steps
        elif self.observation_mode == AZ_HISTORY_MODE:
            self.observation_channels = 2 * self.temporal_num_steps + 1
        else:
            raise ValueError(f"Unknown observation_mode: {self.observation_mode}")

        if self.observation_mode == TEMPORAL_MOVE_HISTORY_MODE:
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

        if self.observation_mode == AZ_HISTORY_MODE:
            self.black_board_history = torch.zeros(
                num_envs,
                self.temporal_num_steps,
                board_size,
                board_size,
                device=self.device,
                dtype=torch.float32,
            )
            self.white_board_history = torch.zeros(
                num_envs,
                self.temporal_num_steps,
                board_size,
                board_size,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self.black_board_history = None
            self.white_board_history = None

        self.observation_spec = CompositeSpec(
            {
                "observation": UnboundedContinuousTensorSpec(
                    device=self.device,
                    shape=[
                        num_envs,
                        self.observation_channels,
                        board_size,
                        board_size,
                    ],
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

    @staticmethod
    def _resolve_observation_mode(
        observation_mode: str | None,
        use_temporal_feature: bool,
    ) -> str:
        if observation_mode is not None:
            mode = str(observation_mode).strip().lower()
            aliases = {
                "legacy": LEGACY_MODE,
                "original": LEGACY_MODE,
                "temporal": TEMPORAL_MOVE_HISTORY_MODE,
                "temporal_move_history": TEMPORAL_MOVE_HISTORY_MODE,
                "move_history": TEMPORAL_MOVE_HISTORY_MODE,
                "az": AZ_HISTORY_MODE,
                "az_history": AZ_HISTORY_MODE,
                "alphago_zero": AZ_HISTORY_MODE,
                "alphazero": AZ_HISTORY_MODE,
            }
            if mode not in aliases:
                raise ValueError(f"Unknown observation_mode: {observation_mode}")
            return aliases[mode]
        return TEMPORAL_MOVE_HISTORY_MODE if use_temporal_feature else LEGACY_MODE

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
        encoded = self.gomoku.get_encoded_board()
        return encoded[:, :2]

    def _get_absolute_board_planes(self) -> tuple[torch.Tensor, torch.Tensor]:
        black = (self.gomoku.board == 1).float()
        white = (self.gomoku.board == -1).float()
        return black, white

    def _get_side_to_move_plane(self) -> torch.Tensor:
        # core.py: turn == 0 means black to move, turn == 1 means white to move
        black_to_move = (self.gomoku.turn == 0).float().view(-1, 1, 1, 1)
        return black_to_move.expand(-1, 1, self.board_size, self.board_size)

    def _build_observation(self) -> torch.Tensor:
        if self.observation_mode == LEGACY_MODE:
            return self.gomoku.get_encoded_board()

        if self.observation_mode == TEMPORAL_MOVE_HISTORY_MODE:
            current_board = self._get_current_board_planes()
            return torch.cat([current_board, self.move_history], dim=1)

        if self.observation_mode == AZ_HISTORY_MODE:
            side_to_move = self._get_side_to_move_plane()
            return torch.cat(
                [self.black_board_history, self.white_board_history, side_to_move],
                dim=1,
            )

        raise RuntimeError(f"Unsupported observation_mode: {self.observation_mode}")

    def _clear_history(self, env_indices: torch.Tensor | None = None) -> None:
        if env_indices is not None and env_indices.numel() == 0:
            return

        if self.move_history is not None:
            if env_indices is None:
                self.move_history.zero_()
            else:
                self.move_history[env_indices] = 0.0

        if self.black_board_history is not None:
            if env_indices is None:
                self.black_board_history.zero_()
                self.white_board_history.zero_()
            else:
                self.black_board_history[env_indices] = 0.0
                self.white_board_history[env_indices] = 0.0

    def _push_last_move_to_history(self, update_mask: torch.Tensor) -> None:
        if self.observation_mode != TEMPORAL_MOVE_HISTORY_MODE or not update_mask.any():
            return

        env_ids = update_mask.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        if self.temporal_num_steps > 1:
            self.move_history[env_ids, 1:] = self.move_history[env_ids, :-1].clone()
        self.move_history[env_ids, 0] = 0.0

        last_move = self.gomoku.last_move[env_ids]
        x = last_move // self.board_size
        y = last_move % self.board_size
        self.move_history[env_ids, 0, x, y] = 1.0

    def _push_current_board_to_history(self, update_mask: torch.Tensor) -> None:
        if self.observation_mode != AZ_HISTORY_MODE or not update_mask.any():
            return

        env_ids = update_mask.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return

        if self.temporal_num_steps > 1:
            self.black_board_history[env_ids, 1:] = self.black_board_history[
                env_ids, :-1
            ].clone()
            self.white_board_history[env_ids, 1:] = self.white_board_history[
                env_ids, :-1
            ].clone()

        black, white = self._get_absolute_board_planes()
        self.black_board_history[env_ids, 0] = black[env_ids]
        self.white_board_history[env_ids, 0] = white[env_ids]

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
            torch.ones_like(action, dtype=torch.bool)
            if env_mask is None
            else env_mask
        )

        episode_len = self.gomoku.move_count + 1
        win, illegal = self.gomoku.step(action=action, env_mask=env_mask)
        assert not illegal.any()

        self._push_last_move_to_history(update_mask=update_mask)
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
