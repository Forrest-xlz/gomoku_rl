import abc
import time
from collections import defaultdict

import torch
from tensordict import TensorDict
from tensordict.nn import InteractionType, set_interaction_type

from gomoku_rl.utils.augment import augment_transition
from gomoku_rl.utils.log import get_log_func
from gomoku_rl.utils.policy import _policy_t

from .env import GomokuEnv


def make_transition(
    tensordict_t_minus_1: TensorDict,
    tensordict_t: TensorDict,
    tensordict_t_plus_1: TensorDict,
) -> TensorDict:
    """
    Constructs a transition tensor dictionary for a two-player game by integrating
    the game state and actions from three consecutive time steps: t-1, t, and t+1.

    The first argument is the state/action frame for the player being trained.
    The third argument is that player's next decision frame after the opponent's
    response.

    Reward convention:
        reward = win(t) - win(t+1)

    Intuition:
        +1 if the player wins after its own move,
        -1 if the opponent wins after the opponent's response,
         0 otherwise.
    """
    # If a player wins at time t, its opponent cannot win immediately after reset.
    reward: torch.Tensor = (
        tensordict_t.get("win").float()
        - tensordict_t_plus_1.get("win").float()
    ).unsqueeze(-1)

    transition: TensorDict = tensordict_t_minus_1.select(
        "observation",
        "action_mask",
        "action",
        "sample_log_prob",
        "state_value",
        strict=False,
    )

    transition.set(
        "next",
        tensordict_t_plus_1.select(
            "observation",
            "action_mask",
            "state_value",
            strict=False,
        ),
    )

    transition.set(("next", "reward"), reward)

    done = tensordict_t_plus_1["done"] | tensordict_t["done"]

    # done: trajectory boundary, can also be set by rollout truncation later.
    # terminated: real episode end only. It must stay False for rollout truncation
    # so GAE can bootstrap from next_state_value at the segment boundary.
    transition.set(("next", "done"), done)
    transition.set(("next", "terminated"), done.clone())
    return transition


def round(
    env: GomokuEnv,
    policy_black: _policy_t,
    policy_white: _policy_t,
    tensordict_t_minus_1: TensorDict,
    tensordict_t: TensorDict,
    return_black_transitions: bool = True,
    return_white_transitions: bool = True,
):
    """
    Executes one black move and one white move.

    Time convention:
        t_minus_1: white decision frame from the previous half-round.
        t        : black decision frame.
        t_plus_1 : after black move, white decision frame.
        t_plus_2 : after white move, next black decision frame.

    Black transition:
        t -> t_plus_2

    White transition:
        t_minus_1 -> t_plus_1

    Important:
        If the environment was reset at t_minus_1, white did not actually make
        a valid action from that frame. Such white transitions are marked invalid
        and filtered out in PPO.learn().
    """
    tensordict_t_plus_1 = env.step_and_maybe_reset(tensordict=tensordict_t)

    with set_interaction_type(type=InteractionType.RANDOM):
        tensordict_t_plus_1 = policy_white(tensordict_t_plus_1)

    if return_white_transitions:
        transition_white = make_transition(
            tensordict_t_minus_1,
            tensordict_t,
            tensordict_t_plus_1,
        )

        invalid: torch.Tensor = tensordict_t_minus_1["done"]
        transition_white["next", "done"] = (
            invalid | transition_white["next", "done"]
        )
        transition_white["next", "terminated"] = (
            invalid | transition_white["next", "terminated"]
        )
        transition_white.set("invalid", invalid)
    else:
        transition_white = None

    # If black wins at t, env_mask prevents an extra white env step.
    tensordict_t_plus_2 = env.step_and_maybe_reset(
        tensordict_t_plus_1,
        env_mask=~tensordict_t_plus_1.get("done"),
    )

    with set_interaction_type(type=InteractionType.RANDOM):
        tensordict_t_plus_2 = policy_black(tensordict_t_plus_2)

    if return_black_transitions:
        transition_black = make_transition(
            tensordict_t,
            tensordict_t_plus_1,
            tensordict_t_plus_2,
        )

        transition_black.set(
            "invalid",
            torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
        )
    else:
        transition_black = None

    return (
        transition_black,
        transition_white,
        tensordict_t_plus_1,
        tensordict_t_plus_2,
    )


def self_play_step(
    env: GomokuEnv,
    policy: _policy_t,
    tensordict_t_minus_1: TensorDict,
    tensordict_t: TensorDict,
):
    """
    Executes a single self-play step in a Gomoku environment using one policy.
    """
    tensordict_t_plus_1 = env.step_and_maybe_reset(tensordict=tensordict_t)

    with set_interaction_type(type=InteractionType.RANDOM):
        tensordict_t_plus_1 = policy(tensordict_t_plus_1)

    transition = make_transition(
        tensordict_t_minus_1,
        tensordict_t,
        tensordict_t_plus_1,
    )

    return (
        transition,
        tensordict_t,
        tensordict_t_plus_1,
    )


class Collector(abc.ABC):
    @abc.abstractmethod
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
        ...

    @abc.abstractmethod
    def reset(self):
        """Resets the collector's internal state."""
        ...


class SelfPlayCollector(Collector):
    def __init__(
        self,
        env: GomokuEnv,
        policy: _policy_t,
        out_device=None,
        augment: bool = False,
    ):
        self._env = env
        self._policy = policy
        self._out_device = out_device or self._env.device
        self._augment = augment

        self._t = None
        self._t_minus_1 = None

    def reset(self):
        self._env.reset()
        self._t = None
        self._t_minus_1 = None

    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))

        tensordicts = []

        start = time.perf_counter()

        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = self._env.reset()

            with set_interaction_type(type=InteractionType.RANDOM):
                self._t_minus_1 = self._policy(self._t_minus_1)

            self._t = self._env.step(self._t_minus_1)

            with set_interaction_type(type=InteractionType.RANDOM):
                self._t = self._policy(self._t)

        for i in range(steps - 1):
            (
                transition,
                self._t_minus_1,
                self._t,
            ) = self_play_step(
                self._env,
                self._policy,
                self._t_minus_1,
                self._t,
            )

            # Truncate the last transition of this rollout segment.
            if i == steps - 2:
                transition["next", "done"] = torch.ones(
                    transition["next", "done"].shape,
                    dtype=torch.bool,
                    device=transition.device,
                )

            if self._augment:
                transition = augment_transition(transition)

            tensordicts.append(transition.to(self._out_device))

        end = time.perf_counter()

        fps = (steps * self._env.num_envs) / (end - start)

        self._env.set_post_step(None)

        tensordicts = torch.stack(tensordicts, dim=-1)

        info.update({"fps": fps})

        return tensordicts, dict(info)


class VersusPlayCollector(Collector):
    def __init__(
        self,
        env: GomokuEnv,
        policy_black: _policy_t,
        policy_white: _policy_t,
        out_device=None,
        augment: bool = False,
    ):
        self._env = env
        self._policy_black = policy_black
        self._policy_white = policy_white
        self._out_device = out_device or self._env.device
        self._augment = augment

        self._t_minus_1 = None
        self._t = None

        # Old behavior:
        #     Skip i == 0 white transition in every rollout.
        #
        # New behavior:
        #     Skip only the first white transition after collector initialization,
        #     because only that one is built from a fake t_minus_1 state.
        #
        # Later rollout boundaries may contain valid white transitions, so they
        # should be kept.
        self._skip_first_white_transition = True

    def reset(self):
        self._env.reset()
        self._t_minus_1 = None
        self._t = None
        self._skip_first_white_transition = True

    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, TensorDict, dict]:
        steps = (steps // 2) * 2

        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))

        blacks = []
        whites = []

        start = time.perf_counter()

        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = self._env.reset()
            self._t = self._env.reset()

            self._t_minus_1.update(
                {
                    "done": torch.ones(
                        self._env.num_envs,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "win": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                }
            )

            with set_interaction_type(type=InteractionType.RANDOM):
                self._t = self._policy_black(self._t)

            self._t.update(
                {
                    "done": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "win": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                }
            )

            # The initialization above creates a fake t_minus_1, so only the
            # first white transition after this point needs to be skipped.
            self._skip_first_white_transition = True

        for i in range(steps // 2):
            (
                transition_black,
                transition_white,
                self._t_minus_1,
                self._t,
            ) = round(
                self._env,
                self._policy_black,
                self._policy_white,
                self._t_minus_1,
                self._t,
            )

            append_white = not self._skip_first_white_transition

            if self._skip_first_white_transition:
                self._skip_first_white_transition = False

            # Truncate the last transition of this rollout segment.
            #
            # This cuts GAE at the rollout boundary. It does not reset the actual
            # environment state stored in self._t_minus_1 / self._t.
            if i == steps // 2 - 1:
                transition_black["next", "done"] = torch.ones(
                    transition_black["next", "done"].shape,
                    dtype=torch.bool,
                    device=transition_black.device,
                )
                transition_white["next", "done"] = torch.ones(
                    transition_white["next", "done"].shape,
                    dtype=torch.bool,
                    device=transition_white.device,
                )

            if self._augment:
                transition_black = augment_transition(transition_black)

                # The skipped first white transition may come from a fake state,
                # so do not augment it.
                if append_white:
                    transition_white = augment_transition(transition_white)

            blacks.append(transition_black.to(self._out_device))

            if append_white:
                whites.append(transition_white.to(self._out_device))

        blacks = torch.stack(blacks, dim=-1) if blacks else None
        whites = torch.stack(whites, dim=-1) if whites else None

        end = time.perf_counter()

        fps = (steps * self._env.num_envs) / (end - start)

        self._env.set_post_step(None)

        info.update({"fps": fps})

        return blacks, whites, dict(info)


class BlackPlayCollector(Collector):
    def __init__(
        self,
        env: GomokuEnv,
        policy_black: _policy_t,
        policy_white: _policy_t,
        out_device=None,
        augment: bool = False,
    ):
        self._env = env
        self._policy_black = policy_black
        self._policy_white = policy_white
        self._out_device = out_device or self._env.device
        self._augment = augment

        self._t_minus_1 = None
        self._t = None

    def reset(self):
        self._env.reset()
        self._t_minus_1 = None
        self._t = None

    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
        steps = (steps // 2) * 2

        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))

        blacks = []

        start = time.perf_counter()

        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = self._env.reset()
            self._t = self._env.reset()

            self._t_minus_1.update(
                {
                    "done": torch.ones(
                        self._env.num_envs,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "win": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                }
            )

            with set_interaction_type(type=InteractionType.RANDOM):
                self._t = self._policy_black(self._t)

            self._t.update(
                {
                    "done": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "win": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                }
            )

        for i in range(steps // 2):
            (
                transition_black,
                transition_white,
                self._t_minus_1,
                self._t,
            ) = round(
                self._env,
                self._policy_black,
                self._policy_white,
                self._t_minus_1,
                self._t,
                return_black_transitions=True,
                return_white_transitions=False,
            )

            # Truncate the last transition of this rollout segment.
            if i == steps // 2 - 1:
                transition_black["next", "done"] = torch.ones(
                    transition_black["next", "done"].shape,
                    dtype=torch.bool,
                    device=transition_black.device,
                )

            if self._augment:
                transition_black = augment_transition(transition_black)

            blacks.append(transition_black.to(self._out_device))

        blacks = torch.stack(blacks, dim=-1) if blacks else None

        end = time.perf_counter()

        fps = (steps * self._env.num_envs) / (end - start)

        self._env.set_post_step(None)

        info.update({"fps": fps})

        return blacks, dict(info)


class WhitePlayCollector(Collector):
    def __init__(
        self,
        env: GomokuEnv,
        policy_black: _policy_t,
        policy_white: _policy_t,
        out_device=None,
        augment: bool = False,
    ):
        self._env = env
        self._policy_black = policy_black
        self._policy_white = policy_white
        self._out_device = out_device or self._env.device
        self._augment = augment

        self._t_minus_1 = None
        self._t = None

        # Same logic as VersusPlayCollector:
        # skip only the first white transition after collector initialization.
        self._skip_first_white_transition = True

    def reset(self):
        self._env.reset()
        self._t_minus_1 = None
        self._t = None
        self._skip_first_white_transition = True

    @torch.no_grad()
    def rollout(self, steps: int) -> tuple[TensorDict, dict]:
        steps = (steps // 2) * 2

        info: defaultdict[str, float] = defaultdict(float)
        self._env.set_post_step(get_log_func(info))

        whites = []

        start = time.perf_counter()

        if self._t_minus_1 is None and self._t is None:
            self._t_minus_1 = self._env.reset()
            self._t = self._env.reset()

            self._t_minus_1.update(
                {
                    "done": torch.ones(
                        self._env.num_envs,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "win": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    # Placeholder. The first white transition is skipped anyway,
                    # but keeping this key avoids missing-key problems if code
                    # inspects the TensorDict before the skip.
                    "action": -torch.ones(
                        self._env.num_envs,
                        dtype=torch.long,
                        device=self._env.device,
                    ),
                }
            )

            with set_interaction_type(type=InteractionType.RANDOM):
                self._t = self._policy_black(self._t)

            self._t.update(
                {
                    "done": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                    "win": torch.zeros(
                        self._env.num_envs,
                        dtype=torch.bool,
                        device=self._env.device,
                    ),
                }
            )

            self._skip_first_white_transition = True

        for i in range(steps // 2):
            (
                transition_black,
                transition_white,
                self._t_minus_1,
                self._t,
            ) = round(
                self._env,
                self._policy_black,
                self._policy_white,
                self._t_minus_1,
                self._t,
                return_black_transitions=False,
                return_white_transitions=True,
            )

            append_white = not self._skip_first_white_transition

            if self._skip_first_white_transition:
                self._skip_first_white_transition = False

            # Truncate the last transition of this rollout segment.
            if i == steps // 2 - 1:
                transition_white["next", "done"] = torch.ones(
                    transition_white["next", "done"].shape,
                    dtype=torch.bool,
                    device=transition_white.device,
                )

            if self._augment:
                if append_white and len(transition_white) > 0:
                    transition_white = augment_transition(transition_white)

            if append_white:
                whites.append(transition_white.to(self._out_device))

        whites = torch.stack(whites, dim=-1) if whites else None

        end = time.perf_counter()

        fps = (steps * self._env.num_envs) / (end - start)

        self._env.set_post_step(None)

        info.update({"fps": fps})

        return whites, dict(info)
