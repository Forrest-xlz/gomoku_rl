import torch
import torch.nn.functional as F


def compute_done(
    board: torch.Tensor,
    kernel_horizontal: torch.Tensor,
    kernel_vertical: torch.Tensor,
    kernel_diagonal: torch.Tensor,
) -> torch.Tensor:
    """Determines if any game has been won in a batch of Gomoku boards."""
    board = board.unsqueeze(1)  # (E,1,B,B)

    output_horizontal = F.conv2d(board, kernel_horizontal)  # (E,1,B-4,B)
    done_horizontal = (output_horizontal.flatten(start_dim=1) > 4.5).any(dim=-1)

    output_vertical = F.conv2d(board, kernel_vertical)  # (E,1,B,B-4)
    done_vertical = (output_vertical.flatten(start_dim=1) > 4.5).any(dim=-1)

    output_diagonal = F.conv2d(board, kernel_diagonal)  # (E,2,B-4,B-4)
    done_diagonal = (output_diagonal.flatten(start_dim=1) > 4.5).any(dim=-1)

    return done_horizontal | done_vertical | done_diagonal


def _compute_line_counts(
    stones: torch.Tensor,
    kernel_horizontal: torch.Tensor,
    kernel_vertical: torch.Tensor,
    kernel_diagonal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generic directional count maps for one occupancy plane.

    Returns:
      h:  (E, H-k+1, W)
      v:  (E, H, W-k+1)
      d1: (E, H-k+1, W-k+1)
      d2: (E, H-k+1, W-k+1)
    """
    E, B, _ = stones.shape
    kh = kernel_horizontal.shape[-2]
    kw = kernel_vertical.shape[-1]

    if stones.numel() == 0 or B < max(kh, kw):
        z_h = torch.zeros((E, max(B - kh + 1, 0), B), device=stones.device, dtype=torch.float)
        z_v = torch.zeros((E, B, max(B - kw + 1, 0)), device=stones.device, dtype=torch.float)
        z_d = torch.zeros(
            (E, max(B - kh + 1, 0), max(B - kh + 1, 0)),
            device=stones.device,
            dtype=torch.float,
        )
        return z_h, z_v, z_d, z_d.clone()

    x = stones.float().unsqueeze(1)  # (E,1,B,B)
    h = F.conv2d(x, kernel_horizontal).squeeze(1)
    v = F.conv2d(x, kernel_vertical).squeeze(1)
    d1 = F.conv2d(x, kernel_diagonal[:1]).squeeze(1)
    d2 = F.conv2d(x, kernel_diagonal[1:2]).squeeze(1)
    return h, v, d1, d2


def _counts_to_immediate_five_mask(
    atk_h: torch.Tensor,
    atk_v: torch.Tensor,
    atk_d1: torch.Tensor,
    atk_d2: torch.Tensor,
    def_h: torch.Tensor,
    def_v: torch.Tensor,
    def_d1: torch.Tensor,
    def_d2: torch.Tensor,
    empty: torch.Tensor,
) -> torch.Tensor:
    """
    placing one stone there creates an immediate five-in-a-row.
    """
    if empty.numel() == 0:
        return torch.zeros_like(empty, dtype=torch.bool)

    out = torch.zeros_like(empty, dtype=torch.bool)

    # 5x1
    valid = (atk_h == 4) & (def_h == 0)
    h, _ = valid.shape[-2:]
    for t in range(5):
        out[:, t:t + h, :] |= valid & empty[:, t:t + h, :]

    # 1x5
    valid = (atk_v == 4) & (def_v == 0)
    _, w = valid.shape[-2:]
    for t in range(5):
        out[:, :, t:t + w] |= valid & empty[:, :, t:t + w]

    # main diagonal
    valid = (atk_d1 == 4) & (def_d1 == 0)
    h, w = valid.shape[-2:]
    for t in range(5):
        out[:, t:t + h, t:t + w] |= valid & empty[:, t:t + h, t:t + w]

    # anti diagonal
    valid = (atk_d2 == 4) & (def_d2 == 0)
    h, w = valid.shape[-2:]
    for t in range(5):
        out[:, t:t + h, 4 - t:4 - t + w] |= (
            valid & empty[:, t:t + h, 4 - t:4 - t + w]
        )

    return out


def _counts_to_four_creation_mask(
    atk_h: torch.Tensor,
    atk_v: torch.Tensor,
    atk_d1: torch.Tensor,
    atk_d2: torch.Tensor,
    def_h: torch.Tensor,
    def_v: torch.Tensor,
    def_d1: torch.Tensor,
    def_d2: torch.Tensor,
    empty: torch.Tensor,
) -> torch.Tensor:
    """
    placing one stone there creates some 4-in-5 window (forcing-four point).
    Local condition: atk5 == 3 and def5 == 0
    """
    if empty.numel() == 0:
        return torch.zeros_like(empty, dtype=torch.bool)

    out = torch.zeros_like(empty, dtype=torch.bool)

    # 5x1
    valid = (atk_h == 3) & (def_h == 0)
    h, _ = valid.shape[-2:]
    for t in range(5):
        out[:, t:t + h, :] |= valid & empty[:, t:t + h, :]

    # 1x5
    valid = (atk_v == 3) & (def_v == 0)
    _, w = valid.shape[-2:]
    for t in range(5):
        out[:, :, t:t + w] |= valid & empty[:, :, t:t + w]

    # main diagonal
    valid = (atk_d1 == 3) & (def_d1 == 0)
    h, w = valid.shape[-2:]
    for t in range(5):
        out[:, t:t + h, t:t + w] |= valid & empty[:, t:t + h, t:t + w]

    # anti diagonal
    valid = (atk_d2 == 3) & (def_d2 == 0)
    h, w = valid.shape[-2:]
    for t in range(5):
        out[:, t:t + h, 4 - t:4 - t + w] |= (
            valid & empty[:, t:t + h, 4 - t:4 - t + w]
        )

    return out


def _counts_to_open_four_creation_mask(
    atk6_h: torch.Tensor,
    atk6_v: torch.Tensor,
    atk6_d1: torch.Tensor,
    atk6_d2: torch.Tensor,
    def6_h: torch.Tensor,
    def6_v: torch.Tensor,
    def6_d1: torch.Tensor,
    def6_d2: torch.Tensor,
    end6_h: torch.Tensor,
    end6_v: torch.Tensor,
    end6_d1: torch.Tensor,
    end6_d2: torch.Tensor,
    empty: torch.Tensor,
) -> torch.Tensor:
    """
    T2: placing one stone there creates a true open four: _XXXX_

    6-cell window condition:
      atk6 == 3, def6 == 0, endpoint-empty-count == 2
    Then the unique empty among the middle 4 positions is the move.
    """
    if empty.numel() == 0:
        return torch.zeros_like(empty, dtype=torch.bool)

    E, B, _ = empty.shape
    if B < 6:
        return torch.zeros_like(empty, dtype=torch.bool)

    out = torch.zeros_like(empty, dtype=torch.bool)

    # 6x1
    valid = (atk6_h == 3) & (def6_h == 0) & (end6_h == 2)
    out[:, 1:B-4, :] |= valid & empty[:, 1:B-4, :]
    out[:, 2:B-3, :] |= valid & empty[:, 2:B-3, :]
    out[:, 3:B-2, :] |= valid & empty[:, 3:B-2, :]
    out[:, 4:B-1, :] |= valid & empty[:, 4:B-1, :]

    # 1x6
    valid = (atk6_v == 3) & (def6_v == 0) & (end6_v == 2)
    out[:, :, 1:B-4] |= valid & empty[:, :, 1:B-4]
    out[:, :, 2:B-3] |= valid & empty[:, :, 2:B-3]
    out[:, :, 3:B-2] |= valid & empty[:, :, 3:B-2]
    out[:, :, 4:B-1] |= valid & empty[:, :, 4:B-1]

    # main diagonal
    valid = (atk6_d1 == 3) & (def6_d1 == 0) & (end6_d1 == 2)
    out[:, 1:B-4, 1:B-4] |= valid & empty[:, 1:B-4, 1:B-4]
    out[:, 2:B-3, 2:B-3] |= valid & empty[:, 2:B-3, 2:B-3]
    out[:, 3:B-2, 3:B-2] |= valid & empty[:, 3:B-2, 3:B-2]
    out[:, 4:B-1, 4:B-1] |= valid & empty[:, 4:B-1, 4:B-1]

    # anti diagonal
    valid = (atk6_d2 == 3) & (def6_d2 == 0) & (end6_d2 == 2)
    out[:, 1:B-4, 4:B-1] |= valid & empty[:, 1:B-4, 4:B-1]
    out[:, 2:B-3, 3:B-2] |= valid & empty[:, 2:B-3, 3:B-2]
    out[:, 3:B-2, 2:B-3] |= valid & empty[:, 3:B-2, 2:B-3]
    out[:, 4:B-1, 1:B-4] |= valid & empty[:, 4:B-1, 1:B-4]

    return out


def _counts_to_open_four_defense_mask(
    atk6_h: torch.Tensor,
    atk6_v: torch.Tensor,
    atk6_d1: torch.Tensor,
    atk6_d2: torch.Tensor,
    def6_h: torch.Tensor,
    def6_v: torch.Tensor,
    def6_d1: torch.Tensor,
    def6_d2: torch.Tensor,
    end6_h: torch.Tensor,
    end6_v: torch.Tensor,
    end6_d1: torch.Tensor,
    end6_d2: torch.Tensor,
    empty: torch.Tensor,
) -> torch.Tensor:
    """
    T3-defense part: ALL defense points against an opponent T2 window.

    For each valid 6-cell threat window:
      atk6 == 3, def6 == 0, endpoint-empty-count == 2

    Keep:
      - both endpoints
      - the unique middle empty (opponent's T2 move itself)
    """
    if empty.numel() == 0:
        return torch.zeros_like(empty, dtype=torch.bool)

    E, B, _ = empty.shape
    if B < 6:
        return torch.zeros_like(empty, dtype=torch.bool)

    out = torch.zeros_like(empty, dtype=torch.bool)

    # 6x1
    valid = (atk6_h == 3) & (def6_h == 0) & (end6_h == 2)
    out[:, 0:B-5, :] |= valid & empty[:, 0:B-5, :]
    out[:, 5:B,   :] |= valid & empty[:, 5:B,   :]
    out[:, 1:B-4, :] |= valid & empty[:, 1:B-4, :]
    out[:, 2:B-3, :] |= valid & empty[:, 2:B-3, :]
    out[:, 3:B-2, :] |= valid & empty[:, 3:B-2, :]
    out[:, 4:B-1, :] |= valid & empty[:, 4:B-1, :]

    # 1x6
    valid = (atk6_v == 3) & (def6_v == 0) & (end6_v == 2)
    out[:, :, 0:B-5] |= valid & empty[:, :, 0:B-5]
    out[:, :, 5:B  ] |= valid & empty[:, :, 5:B  ]
    out[:, :, 1:B-4] |= valid & empty[:, :, 1:B-4]
    out[:, :, 2:B-3] |= valid & empty[:, :, 2:B-3]
    out[:, :, 3:B-2] |= valid & empty[:, :, 3:B-2]
    out[:, :, 4:B-1] |= valid & empty[:, :, 4:B-1]

    # main diagonal
    valid = (atk6_d1 == 3) & (def6_d1 == 0) & (end6_d1 == 2)
    out[:, 0:B-5, 0:B-5] |= valid & empty[:, 0:B-5, 0:B-5]
    out[:, 5:B,   5:B  ] |= valid & empty[:, 5:B,   5:B  ]
    out[:, 1:B-4, 1:B-4] |= valid & empty[:, 1:B-4, 1:B-4]
    out[:, 2:B-3, 2:B-3] |= valid & empty[:, 2:B-3, 2:B-3]
    out[:, 3:B-2, 3:B-2] |= valid & empty[:, 3:B-2, 3:B-2]
    out[:, 4:B-1, 4:B-1] |= valid & empty[:, 4:B-1, 4:B-1]

    # anti diagonal
    valid = (atk6_d2 == 3) & (def6_d2 == 0) & (end6_d2 == 2)
    out[:, 0:B-5, 5:B  ] |= valid & empty[:, 0:B-5, 5:B  ]
    out[:, 5:B,   0:B-5] |= valid & empty[:, 5:B,   0:B-5]
    out[:, 1:B-4, 4:B-1] |= valid & empty[:, 1:B-4, 4:B-1]
    out[:, 2:B-3, 3:B-2] |= valid & empty[:, 2:B-3, 3:B-2]
    out[:, 3:B-2, 2:B-3] |= valid & empty[:, 3:B-2, 2:B-3]
    out[:, 4:B-1, 1:B-4] |= valid & empty[:, 4:B-1, 1:B-4]

    return out


class Gomoku:
    def __init__(
        self,
        num_envs: int,
        board_size: int = 15,
        device=None,
        action_pruning=None,
    ):
        assert num_envs > 0
        assert board_size >= 5

        self.num_envs: int = num_envs
        self.board_size: int = board_size
        self.device = device

        if action_pruning is None:
            action_pruning = {}

        # 只保留一个总开关：enable=true 时，T0/T1/T2/T3 全部启用
        self.action_pruning_enabled = bool(action_pruning.get("enabled", False))

        self.board: torch.Tensor = torch.zeros(
            num_envs,
            self.board_size,
            self.board_size,
            device=self.device,
            dtype=torch.long,
        )
        self.done: torch.Tensor = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )
        self.turn: torch.Tensor = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self.move_count: torch.Tensor = torch.zeros(
            num_envs, dtype=torch.long, device=self.device
        )
        self.last_move: torch.Tensor = -torch.ones(
            num_envs, dtype=torch.long, device=self.device
        )

        # length-5 kernels
        self.kernel_horizontal = (
            torch.tensor([1, 1, 1, 1, 1], device=self.device, dtype=torch.float)
            .unsqueeze(-1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.kernel_vertical = (
            torch.tensor([1, 1, 1, 1, 1], device=self.device, dtype=torch.float)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.kernel_diagonal = torch.tensor(
            [
                [
                    [1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ],
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                ],
            ],
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(1)

        # length-6 sum kernels
        self.kernel_horizontal6 = (
            torch.tensor([1, 1, 1, 1, 1, 1], device=self.device, dtype=torch.float)
            .unsqueeze(-1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.kernel_vertical6 = (
            torch.tensor([1, 1, 1, 1, 1, 1], device=self.device, dtype=torch.float)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.kernel_diagonal6 = torch.tensor(
            [
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                [
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                ],
            ],
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(1)

        # endpoint-empty kernels for 6-cell windows
        self.kernel_horizontal6_end = (
            torch.tensor([1, 0, 0, 0, 0, 1], device=self.device, dtype=torch.float)
            .unsqueeze(-1)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.kernel_vertical6_end = (
            torch.tensor([1, 0, 0, 0, 0, 1], device=self.device, dtype=torch.float)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.kernel_diagonal6_end = torch.tensor(
            [
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                [
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                ],
            ],
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(1)

    def to(self, device):
        self.board.to(device=device)
        self.done.to(device=device)
        self.turn.to(device=device)
        self.move_count.to(device=device)
        self.last_move.to(device=device)
        return self

    def reset(self, env_indices: torch.Tensor | None = None):
        if env_indices is None:
            self.board.zero_()
            self.done.zero_()
            self.turn.zero_()
            self.move_count.zero_()
            self.last_move.fill_(-1)
        else:
            self.board[env_indices] = 0
            self.done[env_indices] = False
            self.turn[env_indices] = 0
            self.move_count[env_indices] = 0
            self.last_move[env_indices] = -1

    def step(
        self, action: torch.Tensor, env_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if env_mask is None:
            env_mask = torch.ones_like(action, dtype=torch.bool)

        board_1d_view = self.board.view(self.num_envs, -1)

        values_on_board = board_1d_view[
            torch.arange(self.num_envs, device=self.device),
            action,
        ]

        nop = (values_on_board != 0) | (~env_mask)
        inc = torch.logical_not(nop).long()
        piece = torch.where(self.turn == 0, 1, -1)

        board_1d_view[
            torch.arange(self.num_envs, device=self.device), action
        ] = torch.where(nop, values_on_board, piece)

        self.move_count = self.move_count + inc

        board_one_side = (self.board == piece.unsqueeze(-1).unsqueeze(-1)).float()

        self.done = compute_done(
            board_one_side,
            self.kernel_horizontal,
            self.kernel_vertical,
            self.kernel_diagonal,
        ) | (self.move_count == self.board_size * self.board_size)

        self.turn = (self.turn + inc) % 2
        self.last_move = torch.where(nop, self.last_move, action)

        return self.done & env_mask, nop & env_mask

    def get_encoded_board(self) -> torch.Tensor:
        piece = torch.where(self.turn == 0, 1, -1).unsqueeze(-1).unsqueeze(-1)

        layer1 = (self.board == piece).float()
        layer2 = (self.board == -piece).float()

        last_x = self.last_move // self.board_size
        last_y = self.last_move % self.board_size

        layer3 = (
            (
                torch.arange(self.board_size, device=self.device).unsqueeze(0)
                == last_x.unsqueeze(-1)
            ).unsqueeze(-1)
        ) & (
            (
                torch.arange(self.board_size, device=self.device).unsqueeze(0)
                == last_y.unsqueeze(-1)
            ).unsqueeze(1)
        )
        layer3 = layer3.float()

        return torch.stack([layer1, layer2, layer3], dim=1)

    def get_action_mask(self) -> torch.Tensor:
        """
        Priority:
        T0) current player immediate five
        T1) block opponent immediate five
        T2) current player creates true open four: _XXXX_
        T3) opponent T2 full defense points ∪ current player's forcing-four points
        fallback) normal legal moves

        Notes:
        - enable=True 时，T0/T1/T2/T3 全部开启
        - 5 格卷积只算 1 次，供 T0/T1/T3 复用
        - 6 格卷积只算 1 次，供 T2/T3 复用
        - end6 卷积只算 1 次，供 T2/T3 复用
        - 若某个 env 在高优先级阶段命中，则不会进入后续低优先级阶段
        - T3 只在“对手真的存在 T2 威胁”时触发
        """
        legal = (self.board == 0)
        if not self.action_pruning_enabled:
            return legal.flatten(start_dim=1)

        piece = torch.where(self.turn == 0, 1, -1).view(-1, 1, 1)
        mine = (self.board == piece)
        opp = (self.board == -piece)

        final_mask = legal.clone()
        remaining = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        # ============================================================
        # Stage A: 5-cell counts, shared by T0 / T1 / T3
        # ============================================================
        active5 = remaining & (self.move_count >= 6)

        active5_ids = None
        active5_local_mc = None
        mine5_h = mine5_v = mine5_d1 = mine5_d2 = None
        opp5_h = opp5_v = opp5_d1 = opp5_d2 = None
        legal5 = None
        id_to_active5 = None

        if active5.any():
            active5_ids = active5.nonzero(as_tuple=False).squeeze(-1)
            active5_local_mc = self.move_count[active5_ids]

            mine5 = mine[active5]
            opp5 = opp[active5]
            legal5 = legal[active5]

            # 5 格卷积：只算一次，后续 T0/T1/T3 全复用
            mine5_h, mine5_v, mine5_d1, mine5_d2 = _compute_line_counts(
                mine5,
                self.kernel_horizontal,
                self.kernel_vertical,
                self.kernel_diagonal,
            )
            opp5_h, opp5_v, opp5_d1, opp5_d2 = _compute_line_counts(
                opp5,
                self.kernel_horizontal,
                self.kernel_vertical,
                self.kernel_diagonal,
            )

            # global env id -> local index in active5 tensors
            id_to_active5 = torch.full(
                (self.num_envs,),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            id_to_active5[active5_ids] = torch.arange(
                active5_ids.shape[0], device=self.device
            )

            # ---------------- T0: current player immediate five ----------------
            has_t0_local = torch.zeros(
                active5_ids.shape[0], dtype=torch.bool, device=self.device
            )

            t0_eligible = active5_local_mc >= 8
            if t0_eligible.any():
                my_mask5 = _counts_to_immediate_five_mask(
                    mine5_h, mine5_v, mine5_d1, mine5_d2,
                    opp5_h, opp5_v, opp5_d1, opp5_d2,
                    legal5,
                )
                has_t0_local = my_mask5.flatten(start_dim=1).any(dim=1) & t0_eligible

                if has_t0_local.any():
                    hit_ids = active5_ids[has_t0_local]
                    final_mask[hit_ids] = my_mask5[has_t0_local]
                    remaining[hit_ids] = False

            # ---------------- T1: block opponent immediate five ----------------
            has_t1_local = torch.zeros(
                active5_ids.shape[0], dtype=torch.bool, device=self.device
            )

            t1_eligible = active5_local_mc >= 7
            if t1_eligible.any():
                opp_mask5 = _counts_to_immediate_five_mask(
                    opp5_h, opp5_v, opp5_d1, opp5_d2,
                    mine5_h, mine5_v, mine5_d1, mine5_d2,
                    legal5,
                )
                has_t1_local = opp_mask5.flatten(start_dim=1).any(dim=1) & t1_eligible
                has_t1_local = has_t1_local & (~has_t0_local)

                if has_t1_local.any():
                    hit_ids = active5_ids[has_t1_local]
                    final_mask[hit_ids] = opp_mask5[has_t1_local]
                    remaining[hit_ids] = False

        # 若高优先级阶段已经把所有 env 都处理完，直接返回
        if not remaining.any():
            return final_mask.flatten(start_dim=1)

        # board_size < 6 时，不存在 T2/T3
        if self.board_size < 6:
            return final_mask.flatten(start_dim=1)

        # ============================================================
        # Stage B: 6-cell counts, shared by T2 / T3
        # ============================================================
        active6 = remaining & (self.move_count >= 6)
        if not active6.any():
            return final_mask.flatten(start_dim=1)

        active6_ids = active6.nonzero(as_tuple=False).squeeze(-1)

        mine6 = mine[active6]
        opp6 = opp[active6]
        legal6 = legal[active6]

        # 6 格卷积：只算一次，供 T2/T3 复用
        mine6_h, mine6_v, mine6_d1, mine6_d2 = _compute_line_counts(
            mine6,
            self.kernel_horizontal6,
            self.kernel_vertical6,
            self.kernel_diagonal6,
        )
        opp6_h, opp6_v, opp6_d1, opp6_d2 = _compute_line_counts(
            opp6,
            self.kernel_horizontal6,
            self.kernel_vertical6,
            self.kernel_diagonal6,
        )

        # end6 卷积：只算一次，供 T2/T3 复用
        end6_h, end6_v, end6_d1, end6_d2 = _compute_line_counts(
            legal6,
            self.kernel_horizontal6_end,
            self.kernel_vertical6_end,
            self.kernel_diagonal6_end,
        )

        # ---------------- T2: current player creates true open four ----------------
        t2_mask = _counts_to_open_four_creation_mask(
            mine6_h, mine6_v, mine6_d1, mine6_d2,
            opp6_h, opp6_v, opp6_d1, opp6_d2,
            end6_h, end6_v, end6_d1, end6_d2,
            legal6,
        )
        has_t2_local = t2_mask.flatten(start_dim=1).any(dim=1)

        if has_t2_local.any():
            hit_ids = active6_ids[has_t2_local]
            final_mask[hit_ids] = t2_mask[has_t2_local]
            remaining[hit_ids] = False

        # 若 T2 已经处理完所有还活跃 env，直接返回
        if not remaining.any():
            return final_mask.flatten(start_dim=1)

        # ---------------- T3: opponent T2 defense ∪ my forcing-four ----------------
        # 只有对手真的有 T2 威胁，才进入 T3
        opp_t2_def_mask = _counts_to_open_four_defense_mask(
            opp6_h, opp6_v, opp6_d1, opp6_d2,
            mine6_h, mine6_v, mine6_d1, mine6_d2,
            end6_h, end6_v, end6_d1, end6_d2,
            legal6,
        )
        has_opp_t2_local = opp_t2_def_mask.flatten(start_dim=1).any(dim=1)
        has_opp_t2_local = has_opp_t2_local & (~has_t2_local)

        if has_opp_t2_local.any():
            # T3 复用 Stage A 的 5 格卷积结果，不重复算
            loc_in_active5 = id_to_active5[active6_ids[has_opp_t2_local]]

            my_four_mask = _counts_to_four_creation_mask(
                mine5_h[loc_in_active5],
                mine5_v[loc_in_active5],
                mine5_d1[loc_in_active5],
                mine5_d2[loc_in_active5],
                opp5_h[loc_in_active5],
                opp5_v[loc_in_active5],
                opp5_d1[loc_in_active5],
                opp5_d2[loc_in_active5],
                legal5[loc_in_active5],
            )

            t3_mask = opp_t2_def_mask[has_opp_t2_local] | my_four_mask
            hit_ids = active6_ids[has_opp_t2_local]
            final_mask[hit_ids] = t3_mask

        return final_mask.flatten(start_dim=1)

    def is_valid(self, action: torch.Tensor) -> torch.Tensor:
        out_of_range = (action < 0) | (
            action >= self.board_size * self.board_size
        )
        x = action // self.board_size
        y = action % self.board_size
        values_on_board = self.board[
            torch.arange(self.num_envs, device=self.device), x, y
        ]
        not_empty = values_on_board != 0
        invalid = out_of_range | not_empty
        return ~invalid