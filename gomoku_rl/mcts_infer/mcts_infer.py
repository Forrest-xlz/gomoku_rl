from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torchrl.data.tensor_specs import (
    BinaryDiscreteTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)

from gomoku_rl.core import Gomoku
from gomoku_rl.policy import Policy, get_policy


TEMPORAL_MOVE_HISTORY_MODE = "temporal_move_history"


def _resolve_temporal_num_steps(cfg: DictConfig) -> int:
    for key in ("infer_temporal_num_steps", "model_temporal_num_steps", "temporal_num_steps", "n"):
        value = cfg.get(key, None)
        if value is not None:
            steps = int(value)
            if steps >= 1:
                return steps
    return 1


def _normalize_recent_moves(
    recent_moves: Optional[list[tuple[int, int]] | tuple[tuple[int, int], ...]],
    latest_move: Optional[tuple[int, int]],
    max_steps: int,
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    if recent_moves is not None:
        for move in recent_moves:
            if move is None:
                continue
            row, col = int(move[0]), int(move[1])
            rc = (row, col)
            if rc in seen:
                continue
            normalized.append(rc)
            seen.add(rc)
            if len(normalized) >= max_steps:
                return tuple(normalized[:max_steps])

    if latest_move is not None and len(normalized) < max_steps:
        row, col = int(latest_move[0]), int(latest_move[1])
        rc = (row, col)
        if rc not in seen:
            normalized.insert(0, rc)

    return tuple(normalized[:max_steps])


def _prepend_recent_move(
    recent_moves: tuple[tuple[int, int], ...],
    move: tuple[int, int],
    max_steps: int,
) -> tuple[tuple[int, int], ...]:
    row, col = int(move[0]), int(move[1])
    return ((row, col), *recent_moves[: max(0, max_steps - 1)])


class Piece(enum.Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


EMPTY = Piece.EMPTY.value
BLACK = Piece.BLACK.value
WHITE = Piece.WHITE.value


def opponent(player: Piece) -> Piece:
    return Piece.WHITE if player == Piece.BLACK else Piece.BLACK

# 通过棋盘上黑白子数量推断当前轮到谁
def infer_current_player(board: np.ndarray) -> Optional[Piece]: 
    black_count = int((board == BLACK).sum())
    white_count = int((board == WHITE).sum())
    if black_count == white_count:
        return Piece.BLACK
    if black_count == white_count + 1:
        return Piece.WHITE
    return None

# 把棋盘展平成一维后，找出所有空位的位置编号。
def legal_actions(board: np.ndarray) -> np.ndarray: 
    return np.flatnonzero(board.reshape(-1) == EMPTY)

# 把一维位置编号转换回二维坐标。
def action_to_coord(action: int, board_size: int) -> tuple[int, int]:
    return action // board_size, action % board_size

# 检查从指定位置开始，沿四个主要方向（水平、垂直、两条对角线）是否有连续五个相同颜色的棋子。
def check_five_from(board: np.ndarray, row: int, col: int, stone: int) -> bool:
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    h, w = board.shape
    for dr, dc in directions:
        count = 1

        r, c = row + dr, col + dc
        while 0 <= r < h and 0 <= c < w and int(board[r, c]) == stone:
            count += 1
            r += dr
            c += dc

        r, c = row - dr, col - dc
        while 0 <= r < h and 0 <= c < w and int(board[r, c]) == stone:
            count += 1
            r -= dr
            c -= dc

        if count >= 5:
            return True
    return False

# 判断在指定位置落子后，是否形成五子连珠或棋盘已满。
def is_terminal_after_move(
    board: np.ndarray,
    row: int,
    col: int,
) -> tuple[bool, float]:
    stone = int(board[row, col])
    if stone not in (BLACK, WHITE):
        return False, 0.0
    if check_five_from(board, row, col, stone):
        return True, 1.0
    if not np.any(board == EMPTY):
        return True, 0.0
    return False, 0.0

# 根据配置创建一个 policy 实例，供 MCTS 引导使用。
def _resolve_model_layout(
    cfg: DictConfig,
    temporal_num_steps: Optional[int] = None,
) -> tuple[int, int]:
    temporal_num_steps = int(temporal_num_steps or _resolve_temporal_num_steps(cfg))
    if temporal_num_steps < 1:
        raise ValueError(f"temporal_num_steps must be >= 1, got {temporal_num_steps}")
    observation_channels = 2 + temporal_num_steps
    return temporal_num_steps, observation_channels



def _infer_layout_from_state_dict(state_dict: dict) -> Optional[tuple[int, int]]:
    in_channels = None
    for key, value in state_dict.items():
        if key.endswith('cnn.weight') and hasattr(value, 'shape') and len(value.shape) == 4:
            in_channels = int(value.shape[1])
            break
    if in_channels is None:
        return None
    if in_channels < 3:
        raise ValueError(f"Unsupported checkpoint input channels: {in_channels}")
    return in_channels - 2, in_channels



def make_model(
    cfg: DictConfig,
    temporal_num_steps: Optional[int] = None,
) -> Policy:
    board_size = int(cfg.board_size)
    temporal_num_steps, observation_channels = _resolve_model_layout(
        cfg,
        temporal_num_steps=temporal_num_steps,
    )

    action_spec = DiscreteTensorSpec(
        board_size * board_size,
        shape=[1],
        device=cfg.device,
    )
    observation_spec = CompositeSpec(
        {
            "observation": UnboundedContinuousTensorSpec(
                device=cfg.device,
                shape=[2, observation_channels, board_size, board_size],
            ),
            "action_mask": BinaryDiscreteTensorSpec(
                n=board_size * board_size,
                device=cfg.device,
                shape=[2, board_size * board_size],
                dtype=torch.bool,
            ),
        },
        shape=[2],
        device=cfg.device,
    )
    return get_policy(
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=action_spec,
        observation_spec=observation_spec,
        device=cfg.device,
    )


# 加载指定路径的 checkpoint，并返回一个 Policy 实例，供 MCTS 引导使用。
def load_policy(
    cfg: DictConfig,
    checkpoint_path: str | Path,
    temporal_num_steps: Optional[int] = None,
) -> tuple[Policy, int]:
    state_dict = torch.load(str(checkpoint_path), map_location=cfg.device)

    inferred_layout = _infer_layout_from_state_dict(state_dict)
    cfg_steps, cfg_channels = _resolve_model_layout(
        cfg,
        temporal_num_steps=temporal_num_steps,
    )

    if inferred_layout is not None:
        inferred_steps, inferred_channels = inferred_layout
        if inferred_channels != cfg_channels:
            logging.warning(
                'Checkpoint input channels (%d) do not match cfg-derived channels (%d). '
                'Auto-switching infer temporal_num_steps to %d.',
                inferred_channels,
                cfg_channels,
                inferred_steps,
            )
            cfg_steps = inferred_steps

    model = make_model(cfg, temporal_num_steps=cfg_steps)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg_steps



# 用 core.Gomoku 复用训练侧的当前棋盘编码与动作掩码逻辑，
# 再在推理侧补上 temporal_move_history 所需的最近 n 手 one-hot 通道。
class CoreStateAdapter:
    def __init__(
        self,
        cfg: DictConfig,
        temporal_num_steps: Optional[int] = None,
    ):
        self.cfg = cfg
        self.board_size = int(cfg.board_size)
        self.device = cfg.device
        self.action_pruning_cfg = cfg.get("action_pruning", None)
        self.temporal_num_steps, _ = _resolve_model_layout(
            cfg,
            temporal_num_steps=temporal_num_steps,
        )
        self.env = Gomoku(
            num_envs=1,
            board_size=self.board_size,
            device=self.device,
            action_pruning=self.action_pruning_cfg,
        )
        self.env.reset()

    def _to_core_board(self, board: np.ndarray) -> torch.Tensor:
        board_tensor = torch.as_tensor(board, dtype=torch.long, device=self.device)
        signed_board = torch.zeros_like(board_tensor)
        signed_board = torch.where(board_tensor == BLACK, torch.ones_like(signed_board), signed_board)
        signed_board = torch.where(board_tensor == WHITE, -torch.ones_like(signed_board), signed_board)
        return signed_board

    @torch.no_grad()
    def sync_state(
        self,
        board: np.ndarray,
        current_player: Piece,
        latest_move: Optional[tuple[int, int]],
    ) -> None:
        board_tensor = self._to_core_board(board)
        self.env.board[0].copy_(board_tensor)
        self.env.turn[0] = 0 if current_player == Piece.BLACK else 1
        self.env.move_count[0] = int(np.count_nonzero(board != EMPTY))
        self.env.done[0] = False

        if latest_move is None:
            self.env.last_move[0] = -1
        else:
            row, col = latest_move
            self.env.last_move[0] = int(row * self.board_size + col)

    @torch.no_grad()
    def _build_temporal_move_history(self, recent_moves: tuple[tuple[int, int], ...]) -> torch.Tensor:
        history = torch.zeros(
            1,
            self.temporal_num_steps,
            self.board_size,
            self.board_size,
            device=self.device,
            dtype=torch.float32,
        )
        for idx, move in enumerate(recent_moves[: self.temporal_num_steps]):
            row, col = int(move[0]), int(move[1])
            if 0 <= row < self.board_size and 0 <= col < self.board_size:
                history[0, idx, row, col] = 1.0
        return history

    @torch.no_grad()
    def build_tensordict(
        self,
        board: np.ndarray,
        current_player: Piece,
        latest_move: Optional[tuple[int, int]],
        recent_moves: Optional[tuple[tuple[int, int], ...]] = None,
    ) -> TensorDict:
        self.sync_state(
            board=board,
            current_player=current_player,
            latest_move=latest_move,
        )

        current_planes = self.env.get_encoded_board()[:, :2]
        history_planes = self._build_temporal_move_history(recent_moves or ())
        observation = torch.cat([current_planes, history_planes], dim=1)

        return TensorDict(
            {
                "observation": observation,
                "action_mask": self.env.get_action_mask(),
            },
            batch_size=1,
            device=self.device,
        )


# 构建神经网络输入，直接复用 core 里的编码与动作掩码逻辑。
def build_model_input(
    board: np.ndarray,
    current_player: Piece,
    latest_move: Optional[tuple[int, int]],
    recent_moves: Optional[tuple[tuple[int, int], ...]],
    state_adapter: CoreStateAdapter,
) -> TensorDict:
    return state_adapter.build_tensordict(
        board=board,
        current_player=current_player,
        latest_move=latest_move,
        recent_moves=recent_moves,
    )


@dataclass
class MCTSConfig:
    enabled: bool = True
    num_simulations: int = 64
    c_puct: float = 1.5

    # 自适应搜索预算：
    # B = B_base * (1 + alpha * (1 - R'))
    # 其中 R' = actual_child.visit_count / max_sibling_visits
    adaptive_num_simulations: bool = True
    adaptive_budget_alpha: float = 1.0
    # 这个版本固定使用 min = base，不单独暴露下界配置
    adaptive_budget_max: int = 256

    # 根节点完成一轮基础搜索后，如果“visit 最优动作”和“Q 最优动作”不一致，
    # 则按小步长继续加搜，直到二者一致或达到额外预算上限。
    extend_on_root_disagreement: bool = True
    disagreement_extra_simulations_ratio: float = 0.5
    disagreement_max_extra_simulations_ratio: float = 1.0


@dataclass
class ReuseStats:
    reused: bool = False
    matched: bool = False
    reused_depth: int = 0

    # R' = actual_child.visit_count / max_sibling_visits
    reuse_ratio: float = 1.0

    # 真实落子在兄弟节点中的 visit 排名（1 表示最高）
    reuse_rank: int = 1

    matched_action: Optional[int] = None
    matched_child_visits: int = 0
    parent_max_child_visits: int = 0


@dataclass
class RootSearchStats:
    visit_best_action: Optional[int] = None
    q_best_action: Optional[int] = None
    visit_best_visits: int = 0
    q_best_visits: int = 0
    visit_best_q: float = 0.0
    q_best_q: float = 0.0
    q_gap: float = 0.0
    disagreement: bool = False
    extension_triggered: bool = False
    extension_rounds: int = 0
    extra_simulations: int = 0


@dataclass
class MCTSNode:
    board: np.ndarray
    to_play: Piece  # 当前节点表示的棋盘状态下，轮到哪个玩家落子
    latest_move: Optional[tuple[int, int]]
    recent_moves: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    prior: float = 1.0  # 从父节点到当前节点的先验概率，通常由神经网络输出提供
    visit_count: int = 0    # 从父节点到当前节点的访问次数，在 MCTS 中用于平衡探索和利用
    value_sum: float = 0.0  # 从父节点到当前节点的累计价值总和，通常在回溯时更新，用于计算平均价值 Q = value_sum / visit_count
    terminal: bool = False  # 当前节点是否为终局节点，如果是终局节点则不再扩展子节点
    terminal_value: float = 0.0 # 如果当前节点是终局节点，则表示该节点的价值
    children: dict[int, "MCTSNode"] = field(default_factory=dict)   

    @property   # 它把一个方法包装成属性访问接口。
    def q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count    # 计算平均价值 Q = value_sum / visit_count


class NeuralMCTSInfer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.single_model: Optional[Policy] = None
        self.black_model: Optional[Policy] = None
        self.white_model: Optional[Policy] = None

        self.single_checkpoint: Optional[str] = None
        self.black_checkpoint: Optional[str] = None
        self.white_checkpoint: Optional[str] = None

        base_sims = int(cfg.get("mcts_num_simulations", 64))
        self.mcts_cfg = MCTSConfig(
            enabled=bool(cfg.get("mcts_infer_enabled", True)),
            num_simulations=base_sims,
            c_puct=float(cfg.get("mcts_c_puct", 1.5)),
            adaptive_num_simulations=bool(cfg.get("mcts_adaptive_num_simulations", True)),
            adaptive_budget_alpha=float(cfg.get("mcts_adaptive_budget_alpha", 1.0)),
            adaptive_budget_max=int(cfg.get("mcts_adaptive_budget_max", max(base_sims, base_sims * 4))),
            extend_on_root_disagreement=bool(cfg.get("mcts_extend_on_root_disagreement", True)),
            disagreement_extra_simulations_ratio=float(cfg.get("mcts_disagreement_extra_simulations_ratio", 0.5)),
            disagreement_max_extra_simulations_ratio=float(cfg.get("mcts_disagreement_max_extra_simulations_ratio", 1.0)),
        )
        self.mcts_reuse_tree = bool(cfg.get("mcts_reuse_tree", True))
        self.temporal_num_steps = _resolve_temporal_num_steps(cfg)
        self.state_adapter = CoreStateAdapter(
            cfg,
            temporal_num_steps=self.temporal_num_steps,
        )
        # 缓存当前搜索树 root
        self._cached_root: Optional[MCTSNode] = None
        self._last_dynamic_num_simulations: int = self.mcts_cfg.num_simulations
        self._last_reuse_stats: ReuseStats = ReuseStats()
        self._last_root_search_stats: RootSearchStats = RootSearchStats()

    def load_from_cfg(self) -> None:
        if self.cfg.get("checkpoint"):
            self.load_single(self.cfg.checkpoint)
        if self.cfg.get("black_checkpoint"):
            self.load_black(self.cfg.black_checkpoint)
        if self.cfg.get("white_checkpoint"):
            self.load_white(self.cfg.white_checkpoint)
        # 重新加载模型的时候重置缓存树
        self.reset_search_tree()


    def _update_infer_layout(self, temporal_num_steps: int) -> None:
        changed = int(temporal_num_steps) != int(self.temporal_num_steps)
        self.temporal_num_steps = int(temporal_num_steps)
        if changed:
            self.state_adapter = CoreStateAdapter(
                self.cfg,
                temporal_num_steps=self.temporal_num_steps,
            )
            self.reset_search_tree()

    def load_single(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = str(checkpoint_path)
        self.single_model, temporal_num_steps = load_policy(self.cfg, checkpoint_path)
        self._update_infer_layout(temporal_num_steps)
        self.single_checkpoint = checkpoint_path
        logging.info(
            "Loaded single checkpoint: %s (temporal_num_steps=%d)",
            checkpoint_path,
            temporal_num_steps,
        )

    def load_black(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = str(checkpoint_path)
        self.black_model, temporal_num_steps = load_policy(self.cfg, checkpoint_path)
        self._update_infer_layout(temporal_num_steps)
        self.black_checkpoint = checkpoint_path
        logging.info(
            "Loaded black checkpoint: %s (temporal_num_steps=%d)",
            checkpoint_path,
            temporal_num_steps,
        )

    def load_white(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = str(checkpoint_path)
        self.white_model, temporal_num_steps = load_policy(self.cfg, checkpoint_path)
        self._update_infer_layout(temporal_num_steps)
        self.white_checkpoint = checkpoint_path
        logging.info(
            "Loaded white checkpoint: %s (temporal_num_steps=%d)",
            checkpoint_path,
            temporal_num_steps,
        )

    # 推理期的模型选择规则：
    # 1) 若 black/white 双模型都已加载，则按当前执棋方切换；
    # 2) 否则若 single_model 已加载，则单双方都退回到 single_model；
    # 3) 若只加载了 black_model 或只加载了 white_model，也把这一份模型当成单模型回退使用。
    def _select_model(self, current_player: Piece) -> Optional[Policy]:
        if self.black_model is not None and self.white_model is not None:
            return self.black_model if current_player == Piece.BLACK else self.white_model
        if self.single_model is not None:
            return self.single_model
        if self.black_model is not None:
            return self.black_model
        if self.white_model is not None:
            return self.white_model
        return None

    def _using_dual_models(self) -> bool:
        return self.black_model is not None and self.white_model is not None
    
    # 重置缓存树
    def reset_search_tree(self) -> None:
        self._cached_root = None
        self._last_dynamic_num_simulations = self.mcts_cfg.num_simulations
        self._last_reuse_stats = ReuseStats()
        self._last_root_search_stats = RootSearchStats()

    def _same_state(
        self,
        node: MCTSNode,
        board: np.ndarray,
        current_player: Piece,
    ) -> bool:
        return node.to_play == current_player and np.array_equal(node.board, board)
    #   判断新棋盘是不是旧棋盘“往前走了几步”得到的
    def _forward_ply_distance(
        self,
        old_board: np.ndarray,
        new_board: np.ndarray,
    ) -> Optional[int]:
        diff = np.argwhere(old_board != new_board)

        if len(diff) == 0:
            return 0

        for row, col in diff:
            old_v = int(old_board[row, col])
            new_v = int(new_board[row, col])

            # 只能从 EMPTY 变成 BLACK / WHITE
            if old_v != EMPTY:
                return None
            if new_v not in (BLACK, WHITE):
                return None

        return int(len(diff))
    # 在旧树中找匹配当前棋盘的 descendant 路径。
    # 返回 [(action_1, node_1), ..., (action_k, node_k)]，不包含 root 自身。
    def _find_descendant_path_by_board(
        self,
        root: MCTSNode,
        board: np.ndarray,
        current_player: Piece,
        max_depth: int,
    ) -> Optional[list[tuple[int, MCTSNode]]]:
        stack: list[tuple[MCTSNode, list[tuple[int, MCTSNode]], int]] = [(root, [], 0)]

        while stack:
            node, path, depth = stack.pop()

            if self._same_state(node, board, current_player):
                return path

            if depth >= max_depth:
                continue

            for action, child in node.children.items():
                stack.append((child, path + [(action, child)], depth + 1))

        return None

    def _build_reuse_stats_from_path(
        self,
        previous_root: MCTSNode,
        matched_path: list[tuple[int, MCTSNode]],
    ) -> ReuseStats:
        # 没有前进到新边，说明当前棋盘就是旧 root，本次不需要因为“对手偏离”加预算。
        if len(matched_path) == 0:
            return ReuseStats(
                reused=True,
                matched=True,
                reused_depth=0,
                reuse_ratio=1.0,
                reuse_rank=1,
                matched_action=None,
                matched_child_visits=previous_root.visit_count,
                parent_max_child_visits=previous_root.visit_count,
            )

        matched_action, matched_node = matched_path[-1]
        parent_node = previous_root if len(matched_path) == 1 else matched_path[-2][1]

        sibling_visits = [child.visit_count for child in parent_node.children.values()]
        parent_max_child_visits = max(sibling_visits) if sibling_visits else 0
        matched_child_visits = matched_node.visit_count

        # 如果这个父节点下所有兄弟都没真正被搜索过，R' 没信息量，回退成 1.0。
        if parent_max_child_visits <= 0:
            reuse_ratio = 1.0
            reuse_rank = 1
        else:
            reuse_ratio = float(matched_child_visits) / float(parent_max_child_visits)
            reuse_ratio = max(0.0, min(1.0, reuse_ratio))
            reuse_rank = 1 + sum(
                1 for child in parent_node.children.values()
                if child.visit_count > matched_child_visits
            )

        return ReuseStats(
            reused=True,
            matched=True,
            reused_depth=len(matched_path),
            reuse_ratio=reuse_ratio,
            reuse_rank=reuse_rank,
            matched_action=matched_action,
            matched_child_visits=matched_child_visits,
            parent_max_child_visits=parent_max_child_visits,
        )

    # 尝试复用旧树中的子树作为新树根，同时返回 R' 相关统计量。
    def _get_reusable_root_with_stats(
        self,
        board: np.ndarray,
        current_player: Piece,
        latest_move: Optional[tuple[int, int]],
        recent_moves: tuple[tuple[int, int], ...],
    ) -> tuple[MCTSNode, ReuseStats]:
        if self._cached_root is not None:
            previous_root = self._cached_root
            ply_gap = self._forward_ply_distance(previous_root.board, board)

            if ply_gap is not None:
                matched_path = self._find_descendant_path_by_board(
                    root=previous_root,
                    board=board,
                    current_player=current_player,
                    max_depth=ply_gap,
                )
                if matched_path is not None:
                    matched = previous_root if len(matched_path) == 0 else matched_path[-1][1]

                    matched.latest_move = latest_move
                    matched.recent_moves = tuple(recent_moves)

                    self._cached_root = matched
                    reuse_stats = self._build_reuse_stats_from_path(previous_root, matched_path)
                    return matched, reuse_stats

        # 复用失败，重新建 root
        root = MCTSNode(
            board=board.copy(),
            to_play=current_player,
            latest_move=latest_move,
            recent_moves=tuple(recent_moves),
        )
        self._cached_root = root
        return root, ReuseStats(
            reused=False,
            matched=False,
            reused_depth=0,
            reuse_ratio=1.0,
            reuse_rank=1,
            matched_action=None,
            matched_child_visits=0,
            parent_max_child_visits=0,
        )

    def _compute_dynamic_num_simulations(self, reuse_stats: ReuseStats) -> int:
        base = max(1, int(self.mcts_cfg.num_simulations))

        if not self.mcts_cfg.adaptive_num_simulations:
            return base

        # 没有成功复用到旧树信息时，不因为 R' 额外加预算。
        if not reuse_stats.reused or not reuse_stats.matched:
            return base

        alpha = max(0.0, float(self.mcts_cfg.adaptive_budget_alpha))
        r = max(0.0, min(1.0, float(reuse_stats.reuse_ratio)))

        sims = int(round(base * (1.0 + alpha * (1.0 - r))))
        lower = base
        upper = max(lower, int(self.mcts_cfg.adaptive_budget_max))
        sims = max(lower, sims)
        sims = min(upper, sims)
        return sims

    def _actor_forward(
        self,
        model: Policy,
        board: np.ndarray,
        current_player: Piece,
        latest_move: Optional[tuple[int, int]],
        recent_moves: tuple[tuple[int, int], ...],
    ) -> TensorDict:
        td = build_model_input(
            board=board,
            current_player=current_player,
            latest_move=latest_move,
            recent_moves=recent_moves,
            state_adapter=self.state_adapter,
        )
        actor_input = td.select("observation", "action_mask", strict=False)
        actor_out = model.actor(actor_input)
        td.update(actor_out)    # 将神经网络输出的动作概率等信息添加到 td 中，供 MCTS 使用
        return td

    def _policy_value(
        self,
        board: np.ndarray,
        current_player: Piece,
        latest_move: Optional[tuple[int, int]],
        recent_moves: tuple[tuple[int, int], ...],
    ) -> tuple[np.ndarray, float, np.ndarray]:
        model = self._select_model(current_player)
        if model is None:
            raise RuntimeError("当前未加载可用于 MCTS 的模型。")
        if not (hasattr(model, "actor") and hasattr(model, "critic")):
            raise RuntimeError("当前 policy 不具备 actor / critic，无法执行神经网络引导 MCTS。")

        with torch.no_grad():
            td = self._actor_forward(
                model=model,
                board=board,
                current_player=current_player,
                latest_move=latest_move,
                recent_moves=recent_moves,
            )

            probs = td["probs"].squeeze(0).float()
            mask = td["action_mask"].squeeze(0).bool()
            probs = probs.masked_fill(~mask, 0.0)

            if float(probs.sum().item()) <= 0:
                probs = mask.float()

            probs = probs / probs.sum() # 归一化概率分布
            allowed_actions = mask.nonzero(as_tuple=False).view(-1).detach().cpu().numpy()

            critic_input = td.select("hidden", "observation", strict=False)
            critic_out = model.critic(critic_input)
            value = float(critic_out["state_value"].view(-1)[0].item())

        return probs.detach().cpu().numpy(), value, allowed_actions  # 这里的probs本身就是已经归一化了

    # 直接使用神经网络输出的动作概率分布，选择概率最高的合法动作作为建议落子位置。
    def _direct_argmax_action(
        self,
        board: np.ndarray,
        current_player: Piece,
        latest_move: Optional[tuple[int, int]],
        recent_moves: tuple[tuple[int, int], ...],
    ) -> int:
        model = self._select_model(current_player)
        if model is None:
            raise RuntimeError("当前未加载可用于 direct infer 的模型。")
        with torch.no_grad():
            td = self._actor_forward(
                model=model,
                board=board,
                current_player=current_player,
                latest_move=latest_move,
                recent_moves=recent_moves,
            )

            probs = td["probs"].squeeze(0).float()
            mask = td["action_mask"].squeeze(0).bool()
            probs = probs.masked_fill(~mask, 0.0)

            if float(probs.sum().item()) <= 0:
                probs = mask.float()

            action = int(torch.argmax(probs).item())
            return action

    #  在 MCTS 中扩展一个节点，生成所有合法子节点，并根据神经网络输出的先验概率进行初始化。如果启用了根节点噪声，则在根节点扩展时添加 Dirichlet 噪声以增加探索。
    def _expand_node(
        self,
        node: MCTSNode,
        priors: np.ndarray,
        allowed_actions: np.ndarray,
    ) -> None:
        actions = allowed_actions.astype(np.int64, copy=False)
        if actions.size == 0:   # 如果没有合法动作了，说明当前节点是一个终局节点，直接标记为 terminal 并设置 terminal_value，然后返回，不再扩展子节点。
            node.terminal = True
            node.terminal_value = 0.0
            return

        local_priors = priors[actions].astype(np.float64)
        assert local_priors.sum() > 0


        board_size = int(self.cfg.board_size)
        stone = BLACK if node.to_play == Piece.BLACK else WHITE # 当前节点表示的棋盘状态下，轮到哪个玩家落子，就用哪个玩家的棋子来扩展子节点

        for action, prior in zip(actions.tolist(), local_priors.tolist()):  # 遍历所有合法动作，根据神经网络输出的先验概率为每个动作创建一个子节点，并将子节点添加到当前节点的 children 字典中；如果某个动作对应的位置已经被占用（理论上不应该发生，因为 legal_actions 已经过滤了），则跳过该动作。
            row, col = action_to_coord(int(action), board_size)
            if int(node.board[row, col]) != EMPTY: 
                assert False, f"尝试扩展一个非法动作：{action} 对应的位置 ({row}, {col}) 已经被占用。"

            next_board = node.board.copy()  
            next_board[row, col] = stone    # 一个child一个当前局面

            terminal, reward_for_player_who_just_moved = is_terminal_after_move(next_board, row, col)
            # 从 child 当前执棋方的视角定义的
            if terminal:
                if reward_for_player_who_just_moved > 0:
                    terminal_value = -1.0
                else:
                    terminal_value = 0.0
            else:
                terminal_value = 0.0

            child = MCTSNode(
                board=next_board,
                to_play=opponent(node.to_play),
                latest_move=(row, col),
                recent_moves=_prepend_recent_move(node.recent_moves, (row, col), self.state_adapter.temporal_num_steps),
                prior=float(prior),
                terminal=terminal,
                terminal_value=terminal_value,
            )
            node.children[int(action)] = child

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        best_action = -1
        best_child: Optional[MCTSNode] = None
        best_score = -1e18

        sqrt_parent_visits = math.sqrt(max(1, node.visit_count))
        c_puct = self.mcts_cfg.c_puct

        for action, child in node.children.items():
            q = -child.q
            u = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        if best_child is None:
            raise RuntimeError("MCTS 选择子节点失败。")
        return best_action, best_child

    def _child_value_from_parent_view(self, child: MCTSNode) -> float:
        return -child.q

    def _best_action_by_visit(self, root: MCTSNode) -> tuple[int, MCTSNode]:
        if not root.children:
            raise RuntimeError("根节点没有合法子节点。")
        return max(
            root.children.items(),
            key=lambda kv: (
                kv[1].visit_count,
                self._child_value_from_parent_view(kv[1]),
                -kv[0],
            ),
        )

    def _best_action_by_value(self, root: MCTSNode) -> tuple[int, MCTSNode]:
        if not root.children:
            raise RuntimeError("根节点没有合法子节点。")

        visited_items = [
            (action, child)
            for action, child in root.children.items()
            if child.visit_count > 0
        ]
        candidates = visited_items if visited_items else list(root.children.items())
        return max(
            candidates,
            key=lambda kv: (
                self._child_value_from_parent_view(kv[1]),
                kv[1].visit_count,
                -kv[0],
            ),
        )

    def _analyze_root_search_stats(self, root: MCTSNode) -> RootSearchStats:
        visit_action, visit_child = self._best_action_by_visit(root)
        q_action, q_child = self._best_action_by_value(root)

        visit_q = self._child_value_from_parent_view(visit_child)
        q_best_q = self._child_value_from_parent_view(q_child)
        q_gap = q_best_q - visit_q

        disagreement = visit_action != q_action

        return RootSearchStats(
            visit_best_action=visit_action,
            q_best_action=q_action,
            visit_best_visits=visit_child.visit_count,
            q_best_visits=q_child.visit_count,
            visit_best_q=visit_q,
            q_best_q=q_best_q,
            q_gap=q_gap,
            disagreement=disagreement,
        )

    def _run_single_simulation(self, root: MCTSNode) -> None:
        node = root
        path = [node]

        while node.children and not node.terminal:
            _action, node = self._select_child(node)
            path.append(node)

        if node.terminal:
            value = node.terminal_value
        else:
            priors, value, allowed_actions = self._policy_value(
                board=node.board,
                current_player=node.to_play,
                latest_move=node.latest_move,
                recent_moves=node.recent_moves,
            )
            self._expand_node(
                node,
                priors,
                allowed_actions,
            )

        self._backpropagate(path, value)

    def _run_simulations(self, root: MCTSNode, num_simulations: int) -> None:
        for _ in range(max(0, int(num_simulations))):
            self._run_single_simulation(root)

    def _maybe_extend_search_on_root_disagreement(
        self,
        root: MCTSNode,
        base_num_simulations: int,
    ) -> RootSearchStats:
        stats = self._analyze_root_search_stats(root)
        if not self.mcts_cfg.extend_on_root_disagreement:
            return stats

        chunk_ratio = max(0.0, float(self.mcts_cfg.disagreement_extra_simulations_ratio))
        max_extra_ratio = max(0.0, float(self.mcts_cfg.disagreement_max_extra_simulations_ratio))
        chunk_size = max(1, int(round(base_num_simulations * chunk_ratio))) if chunk_ratio > 0 else 0
        max_extra = max(0, int(round(base_num_simulations * max_extra_ratio)))

        if chunk_size <= 0 or max_extra <= 0:
            return stats

        extra_used = 0
        extension_rounds = 0

        while stats.disagreement and extra_used < max_extra:
            sims_to_add = min(chunk_size, max_extra - extra_used)
            self._run_simulations(root, sims_to_add)
            extra_used += sims_to_add
            extension_rounds += 1
            stats = self._analyze_root_search_stats(root)

        stats.extension_triggered = extra_used > 0
        stats.extension_rounds = extension_rounds
        stats.extra_simulations = extra_used
        return stats

    def _backpropagate(self, path: list[MCTSNode], value: float) -> None:   # 从叶子节点开始，沿着路径向上回溯，更新每个节点的访问次数和价值总和；value 的符号在每层反转，以反映当前节点的视角。
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _run_mcts_with_reuse(
        self,
        board: np.ndarray,
        current_player: Piece,
        latest_move: Optional[tuple[int, int]],
        recent_moves: tuple[tuple[int, int], ...],
    ) -> tuple[int, MCTSNode]:
        root, reuse_stats = self._get_reusable_root_with_stats(
            board=board,
            current_player=current_player,
            latest_move=latest_move,
            recent_moves=recent_moves,
        )
        self._last_reuse_stats = reuse_stats
        num_simulations = self._compute_dynamic_num_simulations(reuse_stats)
        self._last_dynamic_num_simulations = num_simulations

        # 如果这个 root 还没展开，就先展开
        if not root.terminal and not root.children:
            root_priors, _root_value, root_allowed_actions = self._policy_value(
                board=root.board,
                current_player=root.to_play,
                latest_move=root.latest_move,
                recent_moves=root.recent_moves,
            )
            self._expand_node(
                root,
                root_priors,
                root_allowed_actions,
            )

        self._run_simulations(root, num_simulations)
        root_search_stats = self._maybe_extend_search_on_root_disagreement(root, num_simulations)
        self._last_root_search_stats = root_search_stats
        self._last_dynamic_num_simulations = num_simulations + root_search_stats.extra_simulations

        if not root.children:
            raise RuntimeError("根节点没有合法子节点。")

        # 当前真实状态对应的 root 继续缓存下来
        self._cached_root = root

        best_action, best_child = self._best_action_by_visit(root)
        return best_action, best_child
    
    def _run_mcts(
        self,
        board: np.ndarray,
        current_player: Piece,
        latest_move: Optional[tuple[int, int]],
        recent_moves: tuple[tuple[int, int], ...],
    ) -> tuple[int, MCTSNode]:
        root = MCTSNode(
            board=board.copy(),
            to_play=current_player,
            latest_move=latest_move,
            recent_moves=tuple(recent_moves),
        )

        root_priors, _root_value, root_allowed_actions = self._policy_value(
            board=root.board,
            current_player=root.to_play,
            latest_move=root.latest_move,
            recent_moves=root.recent_moves,
        )
        self._expand_node(
            root,
            root_priors,
            root_allowed_actions,
        )  # root 节点的 prior 由神经网络输出提供；如果启用了根节点噪声，则在扩展 root 时添加 Dirichlet 噪声以增加探索。

        num_simulations = max(1, self.mcts_cfg.num_simulations)
        self._run_simulations(root, num_simulations)
        root_search_stats = self._maybe_extend_search_on_root_disagreement(root, num_simulations)
        self._last_root_search_stats = root_search_stats
        self._last_dynamic_num_simulations = num_simulations + root_search_stats.extra_simulations
        self._last_reuse_stats = ReuseStats()

        if not root.children:
            raise RuntimeError("根节点没有合法子节点。")

        best_action, best_child = self._best_action_by_visit(root)
        return best_action, best_child

    def predict(
        self,
        board: np.ndarray,
        latest_move: Optional[tuple[int, int]],
        recent_moves: Optional[list[tuple[int, int]] | tuple[tuple[int, int], ...]] = None,
    ) -> tuple[Optional[tuple[int, int]], str, Optional[Piece]]:
        current_player = infer_current_player(board)
        if current_player is None:
            return None, "当前棋盘黑白子数量不合法，无法判断轮到谁。", None

        if self._select_model(current_player) is None:
            return None, "当前未加载模型，只完成棋盘识别。", current_player

        recent_moves_tuple = _normalize_recent_moves(
            recent_moves=recent_moves,
            latest_move=latest_move,
            max_steps=self.state_adapter.temporal_num_steps,
        )
        latest_move = recent_moves_tuple[0] if recent_moves_tuple else latest_move

        try:
            if self.mcts_cfg.enabled:   # 如果启用了 MCTS，则使用神经网络引导的 MCTS 来选择动作；否则直接使用神经网络输出概率最高的动作作为建议落子位置。
                if self.mcts_reuse_tree:
                    action, best_child = self._run_mcts_with_reuse(
                        board=board,
                        current_player=current_player,
                        latest_move=latest_move,
                        recent_moves=recent_moves_tuple,
                    )
                else:
                    action, best_child = self._run_mcts(
                        board=board,
                        current_player=current_player,
                        latest_move=latest_move,
                        recent_moves=recent_moves_tuple,
                    )
                row, col = action_to_coord(int(action), int(self.cfg.board_size))
                extra = f"(时序输入最近 {self.state_adapter.temporal_num_steps} 手，当前有效 {len(recent_moves_tuple)} 手）"
                model_mode = "dual-model" if self._using_dual_models() else "single-model-fallback"

                root_search_stats = self._last_root_search_stats
                extend_info = ""
                if root_search_stats.extension_triggered:
                    extend_info = (
                        f", disagree_extend=+{root_search_stats.extra_simulations}"
                        f"/{root_search_stats.extension_rounds}轮"
                    )

                if self.mcts_reuse_tree:
                    actual_sims = self._last_dynamic_num_simulations
                    reuse_stats = self._last_reuse_stats
                    reuse_info = (
                        f"reuse_depth={reuse_stats.reused_depth}, "
                        f"R'={reuse_stats.reuse_ratio:.3f}, "
                        f"rank={reuse_stats.reuse_rank}{extend_info}"
                    )
                    msg = (
                        f"[mcts {actual_sims} sims | {model_mode} | {reuse_info}] "
                        f"轮到{current_player.name.lower()}落子，建议第 {row + 1} 行第 {col + 1} 列；"
                        f"visit={best_child.visit_count}，Q={-best_child.q:.3f}{extra}"
                    )
                else:
                    actual_sims = self._last_dynamic_num_simulations
                    msg = (
                        f"[mcts {actual_sims} sims | {model_mode}{extend_info}] "
                        f"轮到{current_player.name.lower()}落子，建议第 {row + 1} 行第 {col + 1} 列；"
                        f"visit={best_child.visit_count}，Q={-best_child.q:.3f}{extra}"
                    )
                return (row, col), msg, current_player

            action = self._direct_argmax_action(
                board=board,
                current_player=current_player,
                latest_move=latest_move,
                recent_moves=recent_moves_tuple,
            )
            row, col = action_to_coord(int(action), int(self.cfg.board_size))
            extra = f"(时序输入最近 {self.state_adapter.temporal_num_steps} 手，当前有效 {len(recent_moves_tuple)} 手）"
            model_mode = "dual-model" if self._using_dual_models() else "single-model-fallback"
            msg = f"[direct | {model_mode}] 轮到{current_player.name.lower()}落子，建议第 {row + 1} 行第 {col + 1} 列{extra}"
            return (row, col), msg, current_player

        except Exception as exc:
            logging.exception("NeuralMCTSInfer.predict failed")
            return None, f"推理失败：{exc}", current_player



class MCTSManager(NeuralMCTSInfer):
    pass
