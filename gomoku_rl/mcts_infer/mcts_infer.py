from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.parsers.expat import model

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

from gomoku_rl.policy import Policy, get_policy


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
def make_model(cfg: DictConfig) -> Policy:
    board_size = int(cfg.board_size)

    action_spec = DiscreteTensorSpec(
        board_size * board_size,
        shape=[1],
        device=cfg.device,
    )
    observation_spec = CompositeSpec(
        {
            "observation": UnboundedContinuousTensorSpec(
                device=cfg.device,
                shape=[2, 3, board_size, board_size],
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
    return get_policy(  # 返回一个 Policy 实例
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=action_spec,
        observation_spec=observation_spec,
        device=cfg.device,
    )

# 加载指定路径的 checkpoint，并返回一个 Policy 实例，供 MCTS 引导使用。
def load_policy(cfg: DictConfig, checkpoint_path: str | Path) -> Policy:    
    model = make_model(cfg)
    state_dict = torch.load(str(checkpoint_path), map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 构建神经网络输入，包括棋盘状态编码、当前玩家编码、最近一手编码，以及动作掩码。
def build_model_input(
    board: np.ndarray,
    current_player: Piece,
    latest_move: Optional[tuple[int, int]],
    device: str,
) -> TensorDict:
    board_tensor = torch.tensor(board, dtype=torch.long, device=device)
    signed_board = torch.zeros_like(board_tensor)
    signed_board = torch.where(board_tensor == BLACK, torch.ones_like(signed_board), signed_board)  # 黑子用 1 表示
    signed_board = torch.where(board_tensor == WHITE, -torch.ones_like(signed_board), signed_board) # 白子用 -1 表示，空位用 0 表示

    piece_value = 1 if current_player == Piece.BLACK else -1    # 当前玩家的棋子用 1 表示，对手的棋子用 -1 表示，空位用 0 表示
    layer_current = (signed_board == piece_value).float()   # 当前玩家棋子的位置为 1，其他位置为 0
    layer_opponent = (signed_board == -piece_value).float() # 对手棋子的位置为 1，其他位置为 0
    layer_last_move = torch.zeros_like(layer_current)   # 最近一手的位置为 1，其他位置为 0

    if latest_move is not None:
        row, col = latest_move
        layer_last_move[row, col] = 1.0

    observation = torch.stack(
        [layer_current, layer_opponent, layer_last_move],
        dim=0,
    ).unsqueeze(0)
    action_mask = (board_tensor == EMPTY).flatten().unsqueeze(0)    # 可落子位置为 True，其他位置为 False

    return TensorDict(
        {
            "observation": observation,
            "action_mask": action_mask,
        },
        batch_size=1,
    )


@dataclass
class MCTSConfig:
    enabled: bool = True
    num_simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.0
    dirichlet_epsilon: float = 0.0
    temperature: float = 0.0


@dataclass
class MCTSNode:
    board: np.ndarray
    to_play: Piece  # 当前节点表示的棋盘状态下，轮到哪个玩家落子
    latest_move: Optional[tuple[int, int]]
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

        self.mcts_cfg = MCTSConfig(
            enabled=bool(cfg.get("mcts_infer_enabled", True)),
            num_simulations=int(cfg.get("mcts_num_simulations", 64)),
            c_puct=float(cfg.get("mcts_c_puct", 1.5)),
            dirichlet_alpha=float(cfg.get("mcts_dirichlet_alpha", 0.0)),
            dirichlet_epsilon=float(cfg.get("mcts_dirichlet_epsilon", 0.0)),
            temperature=float(cfg.get("mcts_temperature", 0.0)),
        )
        self.mcts_reuse_tree = bool(cfg.get("mcts_reuse_tree", True))
        # 缓存当前搜索树 root
        self._cached_root: Optional[MCTSNode] = None

    def load_from_cfg(self) -> None:
        if self.cfg.get("checkpoint"):
            self.load_single(self.cfg.checkpoint)
        if self.cfg.get("black_checkpoint"):
            self.load_black(self.cfg.black_checkpoint)
        if self.cfg.get("white_checkpoint"):
            self.load_white(self.cfg.white_checkpoint)
        # 重新加载模型的时候重置缓存树
        self.reset_search_tree()


    def load_single(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = str(checkpoint_path)
        self.single_model = load_policy(self.cfg, checkpoint_path)
        self.single_checkpoint = checkpoint_path
        logging.info("Loaded single checkpoint: %s", checkpoint_path)

    def load_black(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = str(checkpoint_path)
        self.black_model = load_policy(self.cfg, checkpoint_path)
        self.black_checkpoint = checkpoint_path
        logging.info("Loaded black checkpoint: %s", checkpoint_path)

    def load_white(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = str(checkpoint_path)
        self.white_model = load_policy(self.cfg, checkpoint_path)
        self.white_checkpoint = checkpoint_path
        logging.info("Loaded white checkpoint: %s", checkpoint_path)

    # 如果同时加载了 single_model 和 black_model/white_model，则优先使用 black_model/white_model；否则使用 single_model。
    def _select_model(self, current_player: Piece) -> Optional[Policy]: 
        if self.black_model is not None and self.white_model is not None:
            return self.black_model if current_player == Piece.BLACK else self.white_model
        return self.single_model
    
    # 重置缓存树
    def reset_search_tree(self) -> None:
        self._cached_root = None

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
    # 在旧树中找匹配当前棋盘的 descendant
    def _find_descendant_by_board(
        self,
        root: MCTSNode,
        board: np.ndarray,
        current_player: Piece,
        max_depth: int,
    ) -> Optional[MCTSNode]:
        stack: list[tuple[MCTSNode, int]] = [(root, 0)]

        while stack:
            node, depth = stack.pop()

            if self._same_state(node, board, current_player):
                return node

            if depth >= max_depth:
                continue

            for child in node.children.values():
                stack.append((child, depth + 1))

        return None
    # 尝试复用旧树中的子树作为新树的根节点，以保留搜索树中的信息；如果新棋盘与旧棋盘之间的差异过大（超过 max_depth），或者找不到匹配的新棋盘的节点，则放弃复用，重新创建一个新的根节点。
    def _get_reusable_root(
        self,
        board: np.ndarray,
        current_player: Piece,
        latest_move: Optional[tuple[int, int]],
    ) -> MCTSNode:
        if self._cached_root is not None:
            ply_gap = self._forward_ply_distance(self._cached_root.board, board)

            if ply_gap is not None:
                matched = self._find_descendant_by_board(
                    root=self._cached_root,
                    board=board,
                    current_player=current_player,
                    max_depth=ply_gap,
                )
                if matched is not None:
                    # 如果外部这次提供了 latest_move，就用新的覆盖一下
                    if latest_move is not None:
                        matched.latest_move = latest_move
                    self._cached_root = matched
                    return matched

        # 复用失败，重新建 root
        root = MCTSNode(
            board=board.copy(),
            to_play=current_player,
            latest_move=latest_move,
        )
        self._cached_root = root
        return root

    def _actor_forward(
        self,
        model: Policy,
        board: np.ndarray,
        current_player: Piece,
        latest_move: Optional[tuple[int, int]],
    ) -> TensorDict:
        td = build_model_input(
            board=board,
            current_player=current_player,
            latest_move=latest_move,
            device=self.cfg.device,
        )
        actor_input = td.select("observation", "action_mask", strict=False)
        actor_out = model.actor(actor_input)
        td.update(actor_out)    # 将神经网络输出的动作概率等信息添加到 td 中，供 MCTS 使用
        return td

    def _policy_value(
        self,
        model: Policy,
        board: np.ndarray,
        current_player: Piece,
        latest_move: Optional[tuple[int, int]],
    ) -> tuple[np.ndarray, float]:
        if not (hasattr(model, "actor") and hasattr(model, "critic")):
            raise RuntimeError("当前 policy 不具备 actor / critic，无法执行神经网络引导 MCTS。")

        with torch.no_grad():
            td = self._actor_forward(
                model=model,
                board=board,
                current_player=current_player,
                latest_move=latest_move,
            )

            probs = td["probs"].squeeze(0).float()
            mask = td["action_mask"].squeeze(0).bool()
            probs = probs.masked_fill(~mask, 0.0)

            if float(probs.sum().item()) <= 0:
                probs = mask.float()

            probs = probs / probs.sum() # 归一化概率分布

            critic_input = td.select("hidden", "observation", strict=False)
            critic_out = model.critic(critic_input) 
            value = float(critic_out["state_value"].view(-1)[0].item())

        return probs.detach().cpu().numpy(), value  # 这里的probs本身就是已经归一化了

    # 直接使用神经网络输出的动作概率分布，选择概率最高的合法动作作为建议落子位置。
    def _direct_argmax_action(
        self,
        model: Policy,
        board: np.ndarray,
        current_player: Piece,
        latest_move: Optional[tuple[int, int]],
    ) -> int:
        with torch.no_grad():
            td = self._actor_forward(
                model=model,
                board=board,
                current_player=current_player,
                latest_move=latest_move,
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
        add_root_noise: bool = False,
    ) -> None:
        actions = legal_actions(node.board)
        if actions.size == 0:   # 如果没有合法动作了，说明当前节点是一个终局节点，直接标记为 terminal 并设置 terminal_value，然后返回，不再扩展子节点。
            node.terminal = True
            node.terminal_value = 0.0
            return

        local_priors = priors[actions].astype(np.float64)
        assert local_priors.sum() > 0

        # 给根节点加dirichlet噪声以增加探索，特别是在搜索初期；dirichlet_alpha 控制噪声的分布，dirichlet_epsilon 原始先验概率 和 噪声 混合的比例
        if add_root_noise and self.mcts_cfg.dirichlet_epsilon > 0.0 and self.mcts_cfg.dirichlet_alpha > 0.0:
            noise = np.random.dirichlet(
                alpha=np.full(len(local_priors), self.mcts_cfg.dirichlet_alpha, dtype=np.float64)
            )
            eps = self.mcts_cfg.dirichlet_epsilon
            local_priors = (1.0 - eps) * local_priors + eps * noise
            local_priors = local_priors / local_priors.sum()

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

    def _backpropagate(self, path: list[MCTSNode], value: float) -> None:   # 从叶子节点开始，沿着路径向上回溯，更新每个节点的访问次数和价值总和；value 的符号在每层反转，以反映当前节点的视角。
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _run_mcts_with_reuse(
        self,
        model: Policy,
        board: np.ndarray,
        current_player: Piece,
        latest_move: Optional[tuple[int, int]],
    ) -> tuple[int, MCTSNode]:
        root = self._get_reusable_root(
            board=board,
            current_player=current_player,
            latest_move=latest_move,
        )

        # 如果这个 root 还没展开，就先展开
        if not root.terminal and not root.children:
            root_priors, _ = self._policy_value(
                model=model,
                board=root.board,
                current_player=root.to_play,
                latest_move=root.latest_move,
            )
            self._expand_node(root, root_priors, add_root_noise=False)

        num_simulations = max(1, self.mcts_cfg.num_simulations)
        for _ in range(num_simulations):
            node = root
            path = [node]

            while node.children and not node.terminal:
                _action, node = self._select_child(node)
                path.append(node)

            if node.terminal:
                value = node.terminal_value
            else:
                priors, value = self._policy_value(
                    model=model,
                    board=node.board,
                    current_player=node.to_play,
                    latest_move=node.latest_move,
                )
                self._expand_node(node, priors, add_root_noise=False)

            self._backpropagate(path, value)

        if not root.children:
            raise RuntimeError("根节点没有合法子节点。")

        # 当前真实状态对应的 root 继续缓存下来
        self._cached_root = root

        temperature = float(self.mcts_cfg.temperature)
        if temperature <= 1e-8:
            best_action, best_child = max(
                root.children.items(),
                key=lambda kv: kv[1].visit_count,
            )
            return best_action, best_child

        actions = np.array(list(root.children.keys()), dtype=np.int64)
        visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float64)
        logits = np.log(np.maximum(visits, 1.0)) / temperature
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        sampled_idx = int(np.random.choice(len(actions), p=probs))
        action = int(actions[sampled_idx])
        return action, root.children[action]
    
    def _run_mcts(
        self,
        model: Policy,
        board: np.ndarray,
        current_player: Piece,
        latest_move: Optional[tuple[int, int]],
    ) -> tuple[int, MCTSNode]:
        root = MCTSNode(
            board=board.copy(),
            to_play=current_player,
            latest_move=latest_move,
        )

        root_priors, _ = self._policy_value(
            model=model,
            board=root.board,
            current_player=root.to_play,
            latest_move=root.latest_move,
        )
        self._expand_node(root, root_priors, add_root_noise=False)  # root 节点的 prior 由神经网络输出提供；如果启用了根节点噪声，则在扩展 root 时添加 Dirichlet 噪声以增加探索。

        num_simulations = max(1, self.mcts_cfg.num_simulations)
        for _ in range(num_simulations):
            node = root
            path = [node]

            while node.children and not node.terminal:
                _action, node = self._select_child(node)
                path.append(node)

            if node.terminal:
                value = node.terminal_value
            else:
                priors, value = self._policy_value(
                    model=model,
                    board=node.board,
                    current_player=node.to_play,
                    latest_move=node.latest_move,
                )
                self._expand_node(node, priors, add_root_noise=False)

            self._backpropagate(path, value)

        if not root.children:
            raise RuntimeError("根节点没有合法子节点。")

        #  如果 temperature很小，选择访问次数最多的动作
        temperature = float(self.mcts_cfg.temperature)  
        if temperature <= 1e-8:
            best_action, best_child = max(
                root.children.items(),
                key=lambda kv: kv[1].visit_count,
            )
            return best_action, best_child
        # 根据访问次数分布采样一个动作，temperature 控制分布的平坦程度；temperature 越高，采样越随机；temperature 越低，越倾向于选择访问次数最多的动作。
        actions = np.array(list(root.children.keys()), dtype=np.int64)
        visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float64)
        logits = np.log(np.maximum(visits, 1.0)) / temperature
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        sampled_idx = int(np.random.choice(len(actions), p=probs))
        action = int(actions[sampled_idx])
        return action, root.children[action]

    def predict(
        self,
        board: np.ndarray,
        latest_move: Optional[tuple[int, int]],
    ) -> tuple[Optional[tuple[int, int]], str, Optional[Piece]]:
        current_player = infer_current_player(board)
        if current_player is None:
            return None, "当前棋盘黑白子数量不合法，无法判断轮到谁。", None

        model = self._select_model(current_player)
        if model is None:
            return None, "当前未加载模型，只完成棋盘识别。", current_player

        if latest_move is None and self.cfg.get("require_last_move", False):
            return None, "模型需要 last_move 通道；请先标记最近一手或等待自动差分。", current_player

        try:
            if self.mcts_cfg.enabled:   # 如果启用了 MCTS，则使用神经网络引导的 MCTS 来选择动作；否则直接使用神经网络输出概率最高的动作作为建议落子位置。
                if self.mcts_reuse_tree:
                    action, best_child = self._run_mcts_with_reuse(
                        model=model,
                        board=board,
                        current_player=current_player,
                        latest_move=latest_move,
                    )
                else:
                    action, best_child = self._run_mcts(
                        model=model,
                        board=board,
                        current_player=current_player,
                        latest_move=latest_move,
                    )
                row, col = action_to_coord(int(action), int(self.cfg.board_size))
                extra = "(未提供最近一手，root 的 last_move 通道置零）" if latest_move is None else ""
                msg = (
                    f"[mcts {self.mcts_cfg.num_simulations} sims] "
                    f"轮到{current_player.name.lower()}落子，建议第 {row + 1} 行第 {col + 1} 列；"
                    f"visit={best_child.visit_count}，Q={-best_child.q:.3f}{extra}"
                )
                return (row, col), msg, current_player

            action = self._direct_argmax_action(
                model=model,
                board=board,
                current_player=current_player,
                latest_move=latest_move,
            )
            row, col = action_to_coord(int(action), int(self.cfg.board_size))
            extra = "(未提供最近一手,last_move 通道置零）" if latest_move is None else ""
            msg = f"[direct] 轮到{current_player.name.lower()}落子，建议第 {row + 1} 行第 {col + 1} 列{extra}"
            return (row, col), msg, current_player

        except Exception as exc:
            logging.exception("NeuralMCTSInfer.predict failed")
            return None, f"推理失败：{exc}", current_player



class MCTSManager(NeuralMCTSInfer):
    pass
