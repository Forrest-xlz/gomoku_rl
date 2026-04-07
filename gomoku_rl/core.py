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
        z_h = torch.zeros(
            (E, max(B - kh + 1, 0), B),
            device=stones.device,
            dtype=torch.float,
        )
        z_v = torch.zeros(
            (E, B, max(B - kw + 1, 0)),
            device=stones.device,
            dtype=torch.float,
        )
        z_d = torch.zeros(
            (E, max(B - kh + 1, 0), max(B - kh + 1, 0)),
            device=stones.device,
            dtype=torch.float,
        )
        return z_h, z_v, z_d, z_d.clone()

    x = stones.float().unsqueeze(1)  # (E,1,B,B)
    h = F.conv2d(x, kernel_horizontal).squeeze(1)
    v = F.conv2d(x, kernel_vertical).squeeze(1)
    d = F.conv2d(x, kernel_diagonal)  # (E,2,...)
    d1 = d[:, 0]
    d2 = d[:, 1]
    return h, v, d1, d2


def _compute_two_plane_counts(
    plane_a: torch.Tensor,
    plane_b: torch.Tensor,
    kernel_horizontal: torch.Tensor,
    kernel_vertical: torch.Tensor,
    kernel_diagonal: torch.Tensor,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """
    Compute directional count maps for two occupancy planes in one grouped-conv pass.

    Returns:
      a_h, a_v, a_d1, a_d2, b_h, b_v, b_d1, b_d2
    """
    E, B, _ = plane_a.shape
    kh = kernel_horizontal.shape[-2]
    kw = kernel_vertical.shape[-1]

    if plane_a.numel() == 0 or B < max(kh, kw):
        z_h = torch.zeros(
            (E, max(B - kh + 1, 0), B),
            device=plane_a.device,
            dtype=torch.float,
        )
        z_v = torch.zeros(
            (E, B, max(B - kw + 1, 0)),
            device=plane_a.device,
            dtype=torch.float,
        )
        z_d = torch.zeros(
            (E, max(B - kh + 1, 0), max(B - kh + 1, 0)),
            device=plane_a.device,
            dtype=torch.float,
        )
        return (
            z_h, z_v, z_d, z_d.clone(),
            z_h.clone(), z_v.clone(), z_d.clone(), z_d.clone(),
        )

    x = torch.stack([plane_a.float(), plane_b.float()], dim=1)  # (E,2,B,B)
    h = F.conv2d(x, kernel_horizontal.repeat(2, 1, 1, 1), groups=2)
    v = F.conv2d(x, kernel_vertical.repeat(2, 1, 1, 1), groups=2)
    d = F.conv2d(x, kernel_diagonal.repeat(2, 1, 1, 1), groups=2)

    return (
        h[:, 0], v[:, 0], d[:, 0], d[:, 1],
        h[:, 1], v[:, 1], d[:, 2], d[:, 3],
    )


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

    valid = (atk_h == 4) & (def_h == 0)
    h, _ = valid.shape[-2:]
    for t in range(5):
        out[:, t:t + h, :] |= valid & empty[:, t:t + h, :]

    valid = (atk_v == 4) & (def_v == 0)
    _, w = valid.shape[-2:]
    for t in range(5):
        out[:, :, t:t + w] |= valid & empty[:, :, t:t + w]

    valid = (atk_d1 == 4) & (def_d1 == 0)
    h, w = valid.shape[-2:]
    for t in range(5):
        out[:, t:t + h, t:t + w] |= valid & empty[:, t:t + h, t:t + w]

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

    valid = (atk_h == 3) & (def_h == 0)
    h, _ = valid.shape[-2:]
    for t in range(5):
        out[:, t:t + h, :] |= valid & empty[:, t:t + h, :]

    valid = (atk_v == 3) & (def_v == 0)
    _, w = valid.shape[-2:]
    for t in range(5):
        out[:, :, t:t + w] |= valid & empty[:, :, t:t + w]

    valid = (atk_d1 == 3) & (def_d1 == 0)
    h, w = valid.shape[-2:]
    for t in range(5):
        out[:, t:t + h, t:t + w] |= valid & empty[:, t:t + h, t:t + w]

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

    valid = (atk6_h == 3) & (def6_h == 0) & (end6_h == 2)
    out[:, 1:B-4, :] |= valid & empty[:, 1:B-4, :]
    out[:, 2:B-3, :] |= valid & empty[:, 2:B-3, :]
    out[:, 3:B-2, :] |= valid & empty[:, 3:B-2, :]
    out[:, 4:B-1, :] |= valid & empty[:, 4:B-1, :]

    valid = (atk6_v == 3) & (def6_v == 0) & (end6_v == 2)
    out[:, :, 1:B-4] |= valid & empty[:, :, 1:B-4]
    out[:, :, 2:B-3] |= valid & empty[:, :, 2:B-3]
    out[:, :, 3:B-2] |= valid & empty[:, :, 3:B-2]
    out[:, :, 4:B-1] |= valid & empty[:, :, 4:B-1]

    valid = (atk6_d1 == 3) & (def6_d1 == 0) & (end6_d1 == 2)
    out[:, 1:B-4, 1:B-4] |= valid & empty[:, 1:B-4, 1:B-4]
    out[:, 2:B-3, 2:B-3] |= valid & empty[:, 2:B-3, 2:B-3]
    out[:, 3:B-2, 3:B-2] |= valid & empty[:, 3:B-2, 3:B-2]
    out[:, 4:B-1, 4:B-1] |= valid & empty[:, 4:B-1, 4:B-1]

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

    valid = (atk6_h == 3) & (def6_h == 0) & (end6_h == 2)
    out[:, 0:B-5, :] |= valid & empty[:, 0:B-5, :]
    out[:, 5:B,   :] |= valid & empty[:, 5:B,   :]
    out[:, 1:B-4, :] |= valid & empty[:, 1:B-4, :]
    out[:, 2:B-3, :] |= valid & empty[:, 2:B-3, :]
    out[:, 3:B-2, :] |= valid & empty[:, 3:B-2, :]
    out[:, 4:B-1, :] |= valid & empty[:, 4:B-1, :]

    valid = (atk6_v == 3) & (def6_v == 0) & (end6_v == 2)
    out[:, :, 0:B-5] |= valid & empty[:, :, 0:B-5]
    out[:, :, 5:B  ] |= valid & empty[:, :, 5:B  ]
    out[:, :, 1:B-4] |= valid & empty[:, :, 1:B-4]
    out[:, :, 2:B-3] |= valid & empty[:, :, 2:B-3]
    out[:, :, 3:B-2] |= valid & empty[:, :, 3:B-2]
    out[:, :, 4:B-1] |= valid & empty[:, :, 4:B-1]

    valid = (atk6_d1 == 3) & (def6_d1 == 0) & (end6_d1 == 2)
    out[:, 0:B-5, 0:B-5] |= valid & empty[:, 0:B-5, 0:B-5]
    out[:, 5:B,   5:B  ] |= valid & empty[:, 5:B,   5:B  ]
    out[:, 1:B-4, 1:B-4] |= valid & empty[:, 1:B-4, 1:B-4]
    out[:, 2:B-3, 2:B-3] |= valid & empty[:, 2:B-3, 2:B-3]
    out[:, 3:B-2, 3:B-2] |= valid & empty[:, 3:B-2, 3:B-2]
    out[:, 4:B-1, 4:B-1] |= valid & empty[:, 4:B-1, 4:B-1]

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

        self.env_ids: torch.Tensor = torch.arange(
            num_envs, device=self.device, dtype=torch.long
        )

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
        self.device = device

        self.board = self.board.to(device=device)
        self.done = self.done.to(device=device)
        self.turn = self.turn.to(device=device)
        self.move_count = self.move_count.to(device=device)
        self.last_move = self.last_move.to(device=device)
        self.env_ids = self.env_ids.to(device=device)

        self.kernel_horizontal = self.kernel_horizontal.to(device=device)
        self.kernel_vertical = self.kernel_vertical.to(device=device)
        self.kernel_diagonal = self.kernel_diagonal.to(device=device)

        self.kernel_horizontal6 = self.kernel_horizontal6.to(device=device)
        self.kernel_vertical6 = self.kernel_vertical6.to(device=device)
        self.kernel_diagonal6 = self.kernel_diagonal6.to(device=device)

        self.kernel_horizontal6_end = self.kernel_horizontal6_end.to(device=device)
        self.kernel_vertical6_end = self.kernel_vertical6_end.to(device=device)
        self.kernel_diagonal6_end = self.kernel_diagonal6_end.to(device=device)
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
        values_on_board = board_1d_view[self.env_ids, action]

        nop = (values_on_board != 0) | (~env_mask)
        inc = torch.logical_not(nop).long()
        piece = torch.where(self.turn == 0, 1, -1)

        board_1d_view[self.env_ids, action] = torch.where(
            nop, values_on_board, piece
        )

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

        board_ids = torch.arange(self.board_size, device=self.device)

        layer3 = (
            (board_ids.unsqueeze(0) == last_x.unsqueeze(-1)).unsqueeze(-1)
        ) & (
            (board_ids.unsqueeze(0) == last_y.unsqueeze(-1)).unsqueeze(1)
        )
        layer3 = layer3.float()

        return torch.stack([layer1, layer2, layer3], dim=1)

    @torch.no_grad()
    def get_action_mask(self) -> torch.Tensor:
        """
        Sync-reduced version:
        - no stage-by-stage Python early-stop on GPU tensors
        - compute local masks on GPU first
        - apply T0 > T1 > T2 > T3 priority once at the end

        Priority:
        T0) current player immediate five
        T1) block opponent immediate five
        T2) current player creates true open four: _XXXX_
        T3) opponent T2 full defense points ∪ current player's forcing-four points
        fallback) normal legal moves
        """
        legal = (self.board == 0)
        if not self.action_pruning_enabled:
            return legal.flatten(start_dim=1)

        piece = torch.where(self.turn == 0, 1, -1).view(-1, 1, 1)
        mine = (self.board == piece)
        opp = (self.board == -piece)

        final_mask = legal.clone()

        # ============================================================
        # Stage A data (5-cell): compute for all envs that can possibly need it.
        # No Python-side early-stop here.
        # ============================================================
        active5 = self.move_count >= 6
        active5_ids = active5.nonzero(as_tuple=False).squeeze(-1)
        active5_local_mc = self.move_count[active5_ids]

        mine5 = mine[active5]
        opp5 = opp[active5]
        legal5 = legal[active5]

        (
            mine5_h, mine5_v, mine5_d1, mine5_d2,
            opp5_h, opp5_v, opp5_d1, opp5_d2,
        ) = _compute_two_plane_counts(
            mine5,
            opp5,
            self.kernel_horizontal,
            self.kernel_vertical,
            self.kernel_diagonal,
        )

        my_mask5 = _counts_to_immediate_five_mask(
            mine5_h, mine5_v, mine5_d1, mine5_d2,
            opp5_h, opp5_v, opp5_d1, opp5_d2,
            legal5,
        )
        opp_mask5 = _counts_to_immediate_five_mask(
            opp5_h, opp5_v, opp5_d1, opp5_d2,
            mine5_h, mine5_v, mine5_d1, mine5_d2,
            legal5,
        )

        hit_t0_local = my_mask5.flatten(start_dim=1).any(dim=1) & (active5_local_mc >= 8)
        hit_t1_local = opp_mask5.flatten(start_dim=1).any(dim=1) & (active5_local_mc >= 7) & (~hit_t0_local)

        t0_mask_global = torch.zeros_like(legal)
        t1_mask_global = torch.zeros_like(legal)
        hit_t0_global = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        hit_t1_global = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        t0_mask_global[active5_ids] = my_mask5
        t1_mask_global[active5_ids] = opp_mask5
        hit_t0_global[active5_ids] = hit_t0_local
        hit_t1_global[active5_ids] = hit_t1_local

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

        # ============================================================
        # Stage B data (6-cell): also compute in bulk, then resolve priority later.
        # ============================================================
        if self.board_size >= 6:
            active6 = self.move_count >= 5
            active6_ids = active6.nonzero(as_tuple=False).squeeze(-1)
            active6_local_mc = self.move_count[active6_ids]

            mine6 = mine[active6]
            opp6 = opp[active6]
            legal6 = legal[active6]

            (
                mine6_h, mine6_v, mine6_d1, mine6_d2,
                opp6_h, opp6_v, opp6_d1, opp6_d2,
            ) = _compute_two_plane_counts(
                mine6,
                opp6,
                self.kernel_horizontal6,
                self.kernel_vertical6,
                self.kernel_diagonal6,
            )

            end6_h, end6_v, end6_d1, end6_d2 = _compute_line_counts(
                legal6,
                self.kernel_horizontal6_end,
                self.kernel_vertical6_end,
                self.kernel_diagonal6_end,
            )

            t2_mask_local = _counts_to_open_four_creation_mask(
                mine6_h, mine6_v, mine6_d1, mine6_d2,
                opp6_h, opp6_v, opp6_d1, opp6_d2,
                end6_h, end6_v, end6_d1, end6_d2,
                legal6,
            )
            hit_t2_local = t2_mask_local.flatten(start_dim=1).any(dim=1) & (active6_local_mc >= 6)

            opp_t2_def_mask_local = _counts_to_open_four_defense_mask(
                opp6_h, opp6_v, opp6_d1, opp6_d2,
                mine6_h, mine6_v, mine6_d1, mine6_d2,
                end6_h, end6_v, end6_d1, end6_d2,
                legal6,
            )
            hit_t3_local = opp_t2_def_mask_local.flatten(start_dim=1).any(dim=1) & (~hit_t2_local)

            t3_mask_local = opp_t2_def_mask_local.clone()

            loc_in_active5 = id_to_active5[active6_ids]
            can_add_my_four = hit_t3_local & (loc_in_active5 >= 0)
            valid_loc = loc_in_active5[can_add_my_four]

            my_four_mask_local = _counts_to_four_creation_mask(
                mine5_h[valid_loc],
                mine5_v[valid_loc],
                mine5_d1[valid_loc],
                mine5_d2[valid_loc],
                opp5_h[valid_loc],
                opp5_v[valid_loc],
                opp5_d1[valid_loc],
                opp5_d2[valid_loc],
                legal5[valid_loc],
            )
            t3_mask_local[can_add_my_four] |= my_four_mask_local

            t2_mask_global = torch.zeros_like(legal)
            t3_mask_global = torch.zeros_like(legal)
            hit_t2_global = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            hit_t3_global = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            t2_mask_global[active6_ids] = t2_mask_local
            t3_mask_global[active6_ids] = t3_mask_local
            hit_t2_global[active6_ids] = hit_t2_local
            hit_t3_global[active6_ids] = hit_t3_local
        else:
            t2_mask_global = torch.zeros_like(legal)
            t3_mask_global = torch.zeros_like(legal)
            hit_t2_global = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            hit_t3_global = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ============================================================
        # Resolve priority once, on GPU.
        # ============================================================
        take_t0 = hit_t0_global
        take_t1 = (~take_t0) & hit_t1_global
        take_t2 = (~take_t0) & (~take_t1) & hit_t2_global
        take_t3 = (~take_t0) & (~take_t1) & (~take_t2) & hit_t3_global

        final_mask[take_t3] = t3_mask_global[take_t3]
        final_mask[take_t2] = t2_mask_global[take_t2]
        final_mask[take_t1] = t1_mask_global[take_t1]
        final_mask[take_t0] = t0_mask_global[take_t0]

        return final_mask.flatten(start_dim=1)

    def is_valid(self, action: torch.Tensor) -> torch.Tensor:
        out_of_range = (action < 0) | (
            action >= self.board_size * self.board_size
        )
        x = action // self.board_size
        y = action % self.board_size
        values_on_board = self.board[self.env_ids, x, y]
        not_empty = values_on_board != 0
        invalid = out_of_range | not_empty
        return ~invalid
