from __future__ import annotations

import math
import random
from typing import Callable, Iterable

import torch
from tensordict import TensorDict

from gomoku_rl.core import Gomoku, compute_done


ActionSelector = Callable[["GomokuSimulator"], list[int]]


class GomokuSimulator:
    def __init__(self, gomoku: Gomoku) -> None:
        self.gomoku = gomoku

    @classmethod
    def empty(cls, board_size: int, device=None) -> "GomokuSimulator":
        gomoku = Gomoku(num_envs=1, board_size=board_size, device=device)
        gomoku.reset()
        return cls(gomoku=gomoku)

    @classmethod
    def from_observation(cls, observation: torch.Tensor) -> "GomokuSimulator":
        if observation.ndim != 3 or observation.shape[0] < 2:
            raise ValueError(
                f"expected observation with shape (C, B, B), got {tuple(observation.shape)}"
            )

        obs = observation.detach()
        board_size = int(obs.shape[-1])
        simulator = cls.empty(board_size=board_size, device=obs.device)

        current = (obs[0] > 0.5).to(torch.long)
        opponent = (obs[1] > 0.5).to(torch.long)
        move_count = int(current.sum().item() + opponent.sum().item())
        turn = move_count % 2

        if turn == 0:
            board = current - opponent
        else:
            board = opponent - current

        simulator.gomoku.board[0] = board
        simulator.gomoku.turn[0] = turn
        simulator.gomoku.move_count[0] = move_count
        simulator.gomoku.done[0] = False

        last_move = -1
        if obs.shape[0] >= 3:
            last_layer = obs[2] > 0.5
            if last_layer.any().item():
                last_move = int(last_layer.flatten().nonzero(as_tuple=False)[0].item())
        simulator.gomoku.last_move[0] = last_move
        return simulator

    def clone(self) -> "GomokuSimulator":
        cloned = GomokuSimulator.empty(
            board_size=self.board_size,
            device=self.gomoku.device,
        )
        cloned.gomoku.board.copy_(self.gomoku.board)
        cloned.gomoku.done.copy_(self.gomoku.done)
        cloned.gomoku.turn.copy_(self.gomoku.turn)
        cloned.gomoku.move_count.copy_(self.gomoku.move_count)
        cloned.gomoku.last_move.copy_(self.gomoku.last_move)
        return cloned

    @property
    def board_size(self) -> int:
        return int(self.gomoku.board_size)

    @property
    def turn(self) -> int:
        return int(self.gomoku.turn[0].item())

    @property
    def current_piece(self) -> int:
        return 1 if self.turn == 0 else -1

    @property
    def move_count(self) -> int:
        return int(self.gomoku.move_count[0].item())

    def legal_actions(self) -> list[int]:
        action_mask = self.gomoku.get_action_mask()[0]
        return action_mask.flatten().nonzero(as_tuple=False).flatten().tolist()

    def step(self, action: int) -> None:
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.gomoku.device)
        done, illegal = self.gomoku.step(action=action_tensor)
        if illegal[0].item():
            raise ValueError(f"illegal action {action}")
        self.gomoku.done[0] = done[0]

    def game_end(self) -> tuple[bool, int]:
        if not bool(self.gomoku.done[0].item()):
            return False, 0

        winner = self._winner()
        return True, winner

    def _winner(self) -> int:
        board = self.gomoku.board
        done_black = compute_done(
            (board == 1).float(),
            self.gomoku.kernel_horizontal,
            self.gomoku.kernel_vertical,
            self.gomoku.kernel_diagonal,
        )
        if bool(done_black[0].item()):
            return 1

        done_white = compute_done(
            (board == -1).float(),
            self.gomoku.kernel_horizontal,
            self.gomoku.kernel_vertical,
            self.gomoku.kernel_diagonal,
        )
        if bool(done_white[0].item()):
            return -1
        return 0


class TreeNode:
    def __init__(self, parent: TreeNode | None, prior_p: float) -> None:
        self.parent = parent
        self.children: dict[int, TreeNode] = {}
        self.n_visits = 0
        self.q = 0.0
        self.u = 0.0
        self.p = float(prior_p)

    def expand(self, action_priors: Iterable[tuple[int, float]]) -> None:
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(parent=self, prior_p=prob)

    def select(self, c_puct: float) -> tuple[int, "TreeNode"]:
        return max(
            self.children.items(),
            key=lambda act_node: act_node[1].get_value(c_puct=c_puct),
        )

    def update(self, leaf_value: float) -> None:
        self.n_visits += 1
        self.q += (leaf_value - self.q) / self.n_visits

    def update_recursive(self, leaf_value: float) -> None:
        if self.parent is not None:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct: float) -> float:
        if self.parent is None:
            self.u = 0.0
            return self.q
        self.u = c_puct * self.p * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.q + self.u

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class PureMCTS:
    def __init__(
        self,
        c_puct: float = 5.0,
        n_playout: int = 1000,
        rollout_limit: int | None = None,
        action_selector: ActionSelector | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.root = TreeNode(parent=None, prior_p=1.0)
        self.c_puct = float(c_puct)
        self.n_playout = int(n_playout)
        self.rollout_limit = rollout_limit
        self.action_selector = action_selector
        self.rng = rng if rng is not None else random.Random()

    def _candidate_actions(self, simulator: GomokuSimulator) -> list[int]:
        legal_actions = simulator.legal_actions()
        if self.action_selector is None:
            return legal_actions

        selected_actions = self.action_selector(simulator)
        selected_set = set(int(action) for action in selected_actions)
        filtered_actions = [action for action in legal_actions if action in selected_set]
        return filtered_actions if filtered_actions else legal_actions

    def _policy_value_fn(
        self,
        simulator: GomokuSimulator,
    ) -> tuple[list[tuple[int, float]], float]:
        legal_actions = self._candidate_actions(simulator)
        if not legal_actions:
            return [], 0.0
        prob = 1.0 / len(legal_actions)
        return [(action, prob) for action in legal_actions], 0.0

    def _playout(self, simulator: GomokuSimulator) -> None:
        node = self.root

        while not node.is_leaf():
            action, node = node.select(c_puct=self.c_puct)
            simulator.step(action)

        ended, winner = simulator.game_end()
        if not ended:
            action_priors, _ = self._policy_value_fn(simulator)
            node.expand(action_priors)
            leaf_value = self._evaluate_rollout(simulator)
        else:
            leaf_value = self._terminal_value(simulator=simulator, winner=winner)

        node.update_recursive(-leaf_value)

    def _terminal_value(self, simulator: GomokuSimulator, winner: int) -> float:
        if winner == 0:
            return 0.0
        current_piece = simulator.current_piece
        return 1.0 if winner == current_piece else -1.0

    def _evaluate_rollout(self, simulator: GomokuSimulator) -> float:
        current_piece = simulator.current_piece
        rollout_steps = 0

        while True:
            ended, winner = simulator.game_end()
            if ended:
                if winner == 0:
                    return 0.0
                return 1.0 if winner == current_piece else -1.0

            legal_actions = self._candidate_actions(simulator)
            if not legal_actions:
                return 0.0

            action = self.rng.choice(legal_actions)
            simulator.step(action)
            rollout_steps += 1

            if self.rollout_limit is not None and rollout_steps >= self.rollout_limit:
                return 0.0

    def get_move(self, simulator: GomokuSimulator) -> int:
        legal_actions = simulator.legal_actions()
        if not legal_actions:
            raise ValueError("no legal actions available")
        if len(legal_actions) == 1:
            return legal_actions[0]

        for _ in range(self.n_playout):
            sim_copy = simulator.clone()
            self._playout(sim_copy)

        action, _ = max(
            self.root.children.items(),
            key=lambda act_node: act_node[1].n_visits,
        )
        return int(action)

    def update_with_move(self, last_move: int) -> None:
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(parent=None, prior_p=1.0)

    def reset(self) -> None:
        self.root = TreeNode(parent=None, prior_p=1.0)


class PureMCTSPlayer:
    def __init__(
        self,
        board_size: int,
        num_simulations: int,
        c_puct: float = 5.0,
        rollout_limit: int | None = None,
        seed: int | None = None,
        action_selector: ActionSelector | None = None,
    ) -> None:
        self.board_size = int(board_size)
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.rollout_limit = rollout_limit
        self.action_selector = action_selector
        self.rng = random.Random(seed)

    def eval(self) -> None:
        return None

    def train(self) -> None:
        return None

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        observation = tensordict.get("observation")
        action_mask = tensordict.get("action_mask")
        device = observation.device

        actions: list[int] = []
        for i in range(observation.shape[0]):
            simulator = GomokuSimulator.from_observation(observation[i])
            action = self._select_action_single(
                simulator=simulator,
                action_mask=action_mask[i],
            )
            actions.append(action)

        action_tensor = torch.tensor(actions, dtype=torch.long, device=device)
        tensordict.update({"action": action_tensor})
        return tensordict

    def _select_action_single(
        self,
        simulator: GomokuSimulator,
        action_mask: torch.Tensor,
    ) -> int:
        legal_actions = action_mask.detach().flatten().nonzero(as_tuple=False)
        if legal_actions.numel() == 0:
            raise ValueError("PureMCTSPlayer received a state with no legal actions")
        if legal_actions.numel() == 1:
            return int(legal_actions[0].item())

        mcts = PureMCTS(
            c_puct=self.c_puct,
            n_playout=self.num_simulations,
            rollout_limit=self.rollout_limit,
            action_selector=self.action_selector,
            rng=self.rng,
        )
        return mcts.get_move(simulator)

    def __repr__(self) -> str:
        return (
            f"PureMCTSPlayer(board_size={self.board_size}, "
            f"num_simulations={self.num_simulations}, c_puct={self.c_puct})"
        )