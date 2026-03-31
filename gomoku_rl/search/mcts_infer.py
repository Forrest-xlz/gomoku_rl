from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch

EMPTY = 0
BLACK = 1
WHITE = 2


class MCTSInferenceResult:
    def __init__(
        self,
        action: Optional[int],
        value: float,
        reused_subtree: bool,
        num_simulations: int,
        root_visit_count: int,
    ) -> None:
        self.action = action
        self.value = float(value)
        self.reused_subtree = bool(reused_subtree)
        self.num_simulations = int(num_simulations)
        self.root_visit_count = int(root_visit_count)


class SearchState:
    def __init__(
        self,
        board: np.ndarray,
        current_player: int,
        latest_move: Optional[int],
    ) -> None:
        self.board = np.asarray(board, dtype=np.int8).copy()
        self.current_player = int(current_player)
        self.latest_move = None if latest_move is None else int(latest_move)
        self.board_size = int(self.board.shape[0])

    def clone(self) -> "SearchState":
        return SearchState(
            board=self.board.copy(),
            current_player=self.current_player,
            latest_move=self.latest_move,
        )

    def legal_actions(self) -> list[int]:
        return np.flatnonzero(self.board.reshape(-1) == EMPTY).astype(np.int64).tolist()

    def play(self, action: int) -> "SearchState":
        row = action // self.board_size
        col = action % self.board_size
        if int(self.board[row, col]) != EMPTY:
            raise ValueError(f"illegal action {action}")
        next_board = self.board.copy()
        next_board[row, col] = self.current_player
        next_player = WHITE if self.current_player == BLACK else BLACK
        return SearchState(
            board=next_board,
            current_player=next_player,
            latest_move=action,
        )

    def terminal_result(self) -> Optional[int]:
        if self.latest_move is not None:
            row = self.latest_move // self.board_size
            col = self.latest_move % self.board_size
            piece = int(self.board[row, col])
            if piece != EMPTY and _has_five_from_move(self.board, row, col, piece):
                return piece
        if not np.any(self.board == EMPTY):
            return EMPTY
        return None

    def same_position(self, other: "SearchState") -> bool:
        return (
            self.current_player == other.current_player
            and self.latest_move == other.latest_move
            and self.board.shape == other.board.shape
            and np.array_equal(self.board, other.board)
        )

    def transition_action_to(self, other: "SearchState") -> Optional[int]:
        if self.board.shape != other.board.shape:
            return None
        if other.current_player != _opponent(self.current_player):
            return None
        diff = np.argwhere(self.board != other.board)
        if diff.shape[0] != 1:
            return None
        row, col = map(int, diff[0])
        if int(self.board[row, col]) != EMPTY:
            return None
        if int(other.board[row, col]) != self.current_player:
            return None
        action = row * self.board_size + col
        applied = self.play(action)
        if not np.array_equal(applied.board, other.board):
            return None
        if other.latest_move is not None and int(other.latest_move) != action:
            return None
        return action


class TreeNode:
    def __init__(self, parent: Optional["TreeNode"], prior: float) -> None:
        self.parent = parent
        self.children: dict[int, TreeNode] = {}
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return len(self.children) > 0

    def expand(self, priors: dict[int, float]) -> None:
        for action, prob in priors.items():
            if action not in self.children:
                self.children[action] = TreeNode(parent=self, prior=prob)

    def select_child(self, c_puct: float) -> tuple[int, "TreeNode"]:
        assert self.children, "select_child called on leaf"
        parent_visits = max(1, self.visit_count)
        best_action = -1
        best_node = None
        best_score = -float("inf")
        for action, child in self.children.items():
            u = c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
            score = child.value() + u
            if score > best_score:
                best_score = score
                best_action = action
                best_node = child
        assert best_node is not None
        return best_action, best_node

    def backpropagate(self, leaf_value: float) -> None:
        node: Optional[TreeNode] = self
        value = float(leaf_value)
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent

    def detach(self) -> "TreeNode":
        self.parent = None
        return self


class PolicyValueEvaluator:
    def __init__(
        self,
        build_input_fn: Callable[[np.ndarray, int, Optional[tuple[int, int]], str], "torch.Tensor"],
        select_model_fn: Callable[[int], Optional[object]],
        device: str,
    ) -> None:
        self.build_input_fn = build_input_fn
        self.select_model_fn = select_model_fn
        self.device = device

    def evaluate(self, state: SearchState) -> tuple[dict[int, float], float]:
        model = self.select_model_fn(state.current_player)
        legal_actions = state.legal_actions()
        if model is None:
            if not legal_actions:
                return {}, 0.0
            uniform = 1.0 / len(legal_actions)
            return {action: uniform for action in legal_actions}, 0.0

        latest_move_tuple = None
        if state.latest_move is not None:
            latest_move_tuple = (
                state.latest_move // state.board_size,
                state.latest_move % state.board_size,
            )
        td = self.build_input_fn(
            state.board,
            state.current_player,
            latest_move_tuple,
            self.device,
        )
        with torch.no_grad():
            actor_input = td.select("observation", "action_mask", strict=False)
            actor_output = model.actor(actor_input)
            probs = actor_output.get("probs", None)
            if probs is None:
                action_mask = td.get("action_mask").squeeze(0)
                probs = action_mask.float()
                probs = probs / probs.sum().clamp_min(1.0)
            probs = probs.squeeze(0).detach().cpu()

            value = 0.0
            critic = getattr(model, "critic", None)
            if critic is not None:
                critic_input = actor_output.select("hidden", "observation", strict=False)
                critic_output = critic(critic_input)
                if "state_value" in critic_output.keys():
                    value_tensor = critic_output["state_value"].reshape(-1)
                    if value_tensor.numel() > 0:
                        value = float(value_tensor[0].item())

        priors: dict[int, float] = {}
        total = 0.0
        for action in legal_actions:
            prob = float(probs[action].item())
            if prob > 0.0:
                priors[action] = prob
                total += prob
        if total <= 0.0:
            if not legal_actions:
                return {}, value
            uniform = 1.0 / len(legal_actions)
            return {action: uniform for action in legal_actions}, value
        inv_total = 1.0 / total
        for action in list(priors.keys()):
            priors[action] *= inv_total
        return priors, value


class MCTSInferenceEngine:
    def __init__(
        self,
        board_size: int,
        build_input_fn: Callable[[np.ndarray, int, Optional[tuple[int, int]], str], object],
        select_model_fn: Callable[[int], Optional[object]],
        device: str = "cpu",
        num_simulations: int = 64,
        c_puct: float = 1.5,
        reuse_subtree: bool = True,
    ) -> None:
        self.board_size = int(board_size)
        self.evaluator = PolicyValueEvaluator(
            build_input_fn=build_input_fn,
            select_model_fn=select_model_fn,
            device=device,
        )
        self.num_simulations = int(num_simulations)
        self.c_puct = float(c_puct)
        self.reuse_subtree = bool(reuse_subtree)
        self.root = TreeNode(parent=None, prior=1.0)
        self.root_state: Optional[SearchState] = None
        self.anticipated_root: Optional[TreeNode] = None
        self.anticipated_state: Optional[SearchState] = None

    def reset(self) -> None:
        self.root = TreeNode(parent=None, prior=1.0)
        self.root_state = None
        self.anticipated_root = None
        self.anticipated_state = None

    def observe(
        self,
        board: np.ndarray,
        current_player: int,
        latest_move: Optional[tuple[int, int]],
    ) -> bool:
        state = self._make_state(board, current_player, latest_move)
        return self._sync_to_observed_state(state)

    def suggest(
        self,
        board: np.ndarray,
        current_player: int,
        latest_move: Optional[tuple[int, int]],
    ) -> MCTSInferenceResult:
        state = self._make_state(board, current_player, latest_move)
        reused = self._sync_to_observed_state(state)

        terminal = state.terminal_result()
        if terminal is not None:
            return MCTSInferenceResult(
                action=None,
                value=0.0,
                reused_subtree=reused,
                num_simulations=0,
                root_visit_count=self.root.visit_count,
            )

        if not self.root.expanded():
            priors, _ = self.evaluator.evaluate(state)
            self.root.expand(priors)

        for _ in range(self.num_simulations):
            self._run_single_simulation(state)

        best_action = self._select_action_from_root()
        best_child = self.root.children.get(best_action)
        value = 0.0 if best_child is None else best_child.value()

        if self.reuse_subtree and best_action is not None and best_child is not None:
            self.anticipated_root = best_child.detach()
            self.anticipated_state = state.play(best_action)
        else:
            self.anticipated_root = None
            self.anticipated_state = None

        return MCTSInferenceResult(
            action=best_action,
            value=value,
            reused_subtree=reused,
            num_simulations=self.num_simulations,
            root_visit_count=self.root.visit_count,
        )

    def _make_state(
        self,
        board: np.ndarray,
        current_player: int,
        latest_move: Optional[tuple[int, int]],
    ) -> SearchState:
        latest_move_index = None
        if latest_move is not None:
            latest_move_index = int(latest_move[0]) * self.board_size + int(latest_move[1])
        return SearchState(
            board=np.asarray(board, dtype=np.int8),
            current_player=int(current_player),
            latest_move=latest_move_index,
        )

    def _sync_to_observed_state(self, observed: SearchState) -> bool:
        reused = False
        if self.root_state is None:
            self.root = TreeNode(parent=None, prior=1.0)
            self.root_state = observed
            self.anticipated_root = None
            self.anticipated_state = None
            return reused

        if observed.same_position(self.root_state):
            return True

        if self.anticipated_state is not None and self.anticipated_root is not None:
            if observed.same_position(self.anticipated_state):
                self.root = self.anticipated_root.detach()
                self.root_state = self.anticipated_state
                self.anticipated_root = None
                self.anticipated_state = None
                return True

            anticipated_action = self.anticipated_state.transition_action_to(observed)
            if anticipated_action is not None:
                self.root = self.anticipated_root.detach()
                self.root_state = self.anticipated_state
                self._advance_root(anticipated_action)
                self.root_state = observed
                self.anticipated_root = None
                self.anticipated_state = None
                return True

        direct_action = self.root_state.transition_action_to(observed)
        if direct_action is not None:
            self._advance_root(direct_action)
            self.root_state = observed
            self.anticipated_root = None
            self.anticipated_state = None
            return True

        self.root = TreeNode(parent=None, prior=1.0)
        self.root_state = observed
        self.anticipated_root = None
        self.anticipated_state = None
        return reused

    def _advance_root(self, action: int) -> None:
        child = self.root.children.get(action)
        if child is None:
            self.root = TreeNode(parent=None, prior=1.0)
        else:
            self.root = child.detach()

    def _run_single_simulation(self, state: SearchState) -> None:
        node = self.root
        path_state = state

        while node.expanded():
            action, node = node.select_child(self.c_puct)
            path_state = path_state.play(action)
            terminal = path_state.terminal_result()
            if terminal is not None:
                leaf_value = _terminal_value_for_player(terminal, path_state.current_player)
                node.backpropagate(leaf_value)
                return

        terminal = path_state.terminal_result()
        if terminal is not None:
            leaf_value = _terminal_value_for_player(terminal, path_state.current_player)
            node.backpropagate(leaf_value)
            return

        priors, value = self.evaluator.evaluate(path_state)
        node.expand(priors)
        node.backpropagate(value)

    def _select_action_from_root(self) -> Optional[int]:
        if not self.root.children:
            return None
        best_action = None
        best_visits = -1
        for action, child in self.root.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_action = action
        return best_action


def _opponent(piece: int) -> int:
    return WHITE if piece == BLACK else BLACK


def _terminal_value_for_player(winner: int, player_to_move: int) -> float:
    if winner == EMPTY:
        return 0.0
    return 1.0 if winner == player_to_move else -1.0


def _has_five_from_move(board: np.ndarray, row: int, col: int, piece: int) -> bool:
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    size = int(board.shape[0])
    for dr, dc in directions:
        count = 1
        r, c = row + dr, col + dc
        while 0 <= r < size and 0 <= c < size and int(board[r, c]) == piece:
            count += 1
            r += dr
            c += dc
        r, c = row - dr, col - dc
        while 0 <= r < size and 0 <= c < size and int(board[r, c]) == piece:
            count += 1
            r -= dr
            c -= dc
        if count >= 5:
            return True
    return False
