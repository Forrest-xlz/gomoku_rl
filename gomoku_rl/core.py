import torch
import torch.nn.functional as F


def compute_done(
    board: torch.Tensor,
    kernel_horizontal: torch.Tensor,
    kernel_vertical: torch.Tensor,
    kernel_diagonal: torch.Tensor,
) -> torch.Tensor:
    """Determines if any game has been won in a batch of Gomoku boards.

    Checks for a winning sequence of stones horizontally, vertically, and diagonally.

    Args:
        board (torch.Tensor): The game boards, shaped (E, B, B), with E being the number of environments,
                              and B being the board size. Values are 0 (empty), 1 (black stone), or -1 (white stone).
        kernel_horizontal (torch.Tensor): Horizontal detection kernel, shaped (1, 1, 5, 1).
        kernel_vertical (torch.Tensor): Vertical detection kernel, shaped (1, 1, 1, 5).
        kernel_diagonal (torch.Tensor): Diagonal detection kernels, shaped (2, 1, 5, 5), for both diagonals.

    Returns:
        torch.Tensor: Boolean tensor shaped (E,), indicating if the game is won (True) in each environment.
    """

    board = board.unsqueeze(1)  # (E,1,B,B)

    output_horizontal = F.conv2d(
        input=board, weight=kernel_horizontal)  # (E,1,B-4,B)

    done_horizontal = (output_horizontal.flatten(
        start_dim=1) > 4.5).any(dim=-1)  # (E,)

    output_vertical = F.conv2d(
        input=board, weight=kernel_vertical)  # (E,1,B,B-4)

    done_vertical = (output_vertical.flatten(
        start_dim=1) > 4.5).any(dim=-1)  # (E,)

    output_diagonal = F.conv2d(
        input=board, weight=kernel_diagonal)  # (E,2,B-4,B-4)

    done_diagonal = (output_diagonal.flatten(
        start_dim=1) > 4.5).any(dim=-1)  # (E,)

    done = done_horizontal | done_vertical | done_diagonal

    return done


def _compute_immediate_five_mask(
    atk: torch.Tensor,
    deff: torch.Tensor,
    empty: torch.Tensor,
    kernel_horizontal: torch.Tensor,
    kernel_vertical: torch.Tensor,
    kernel_diagonal: torch.Tensor,
) -> torch.Tensor:
    """
    Returns a boolean mask of shape (E, B, B) where True means:
    placing one stone for `atk` there creates an immediate five-in-a-row.
    """
    if atk.numel() == 0:
        return torch.zeros_like(empty, dtype=torch.bool)

    atk4 = atk.float().unsqueeze(1)   # (E,1,B,B)
    def4 = deff.float().unsqueeze(1)  # (E,1,B,B)
    out = torch.zeros_like(empty, dtype=torch.bool)

    # kernel_horizontal: (5,1) window -> output (E, B-4, B)
    atk_cnt = F.conv2d(atk4, kernel_horizontal).squeeze(1)
    def_cnt = F.conv2d(def4, kernel_horizontal).squeeze(1)
    win_seg = (atk_cnt == 4) & (def_cnt == 0)
    if win_seg.any():
        h, w = win_seg.shape[-2:]
        for t in range(5):
            out[:, t:t + h, :] |= win_seg & empty[:, t:t + h, :]

    # kernel_vertical: (1,5) window -> output (E, B, B-4)
    atk_cnt = F.conv2d(atk4, kernel_vertical).squeeze(1)
    def_cnt = F.conv2d(def4, kernel_vertical).squeeze(1)
    win_seg = (atk_cnt == 4) & (def_cnt == 0)
    if win_seg.any():
        h, w = win_seg.shape[-2:]
        for t in range(5):
            out[:, :, t:t + w] |= win_seg & empty[:, :, t:t + w]

    # Main diagonal
    atk_cnt = F.conv2d(atk4, kernel_diagonal[:1]).squeeze(1)
    def_cnt = F.conv2d(def4, kernel_diagonal[:1]).squeeze(1)
    win_seg = (atk_cnt == 4) & (def_cnt == 0)
    if win_seg.any():
        h, w = win_seg.shape[-2:]
        for t in range(5):
            out[:, t:t + h, t:t + w] |= win_seg & empty[:, t:t + h, t:t + w]

    # Anti diagonal
    atk_cnt = F.conv2d(atk4, kernel_diagonal[1:2]).squeeze(1)
    def_cnt = F.conv2d(def4, kernel_diagonal[1:2]).squeeze(1)
    win_seg = (atk_cnt == 4) & (def_cnt == 0)
    if win_seg.any():
        h, w = win_seg.shape[-2:]
        for t in range(5):
            out[:, t:t + h, 4 - t:4 - t + w] |= (
                win_seg & empty[:, t:t + h, 4 - t:4 - t + w]
            )

    return out


class Gomoku:
    def __init__(
        self,
        num_envs: int,
        board_size: int = 15,
        device=None,
        action_pruning=None,
    ):
        """Initializes a batch of parallel Gomoku game environments.

        Args:
            num_envs (int): Number of parallel game environments.
            board_size (int, optional): Side length of the square game board. Defaults to 15.
            device: Torch device on which the tensors are allocated. Defaults to None (CPU).
            action_pruning: Optional config/dict with keys:
                enabled, self_win4, block_opp_win4
        """
        assert num_envs > 0
        assert board_size >= 5

        self.num_envs: int = num_envs
        self.board_size: int = board_size
        self.device = device

        if action_pruning is None:
            action_pruning = {}

        self.action_pruning_enabled = bool(action_pruning.get("enabled", False))
        self.action_pruning_self_win4 = bool(action_pruning.get("self_win4", True))
        self.action_pruning_block_opp_win4 = bool(
            action_pruning.get("block_opp_win4", True)
        )

        # board 0 empty 1 black -1 white
        self.board: torch.Tensor = torch.zeros(
            num_envs,
            self.board_size,
            self.board_size,
            device=self.device,
            dtype=torch.long,
        )  # (E,B,B)
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

        self.kernel_horizontal = (
            torch.tensor([1, 1, 1, 1, 1], device=self.device, dtype=torch.float)
            .unsqueeze(-1)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1,1,5,1)

        self.kernel_vertical = (
            torch.tensor([1, 1, 1, 1, 1], device=self.device, dtype=torch.float)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1,1,1,5)

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
        ).unsqueeze(1)  # (2,1,5,5)

    def to(self, device):
        """Transfers all internal tensors to the specified device."""
        self.board.to(device=device)
        self.done.to(device=device)
        self.turn.to(device=device)
        self.move_count.to(device=device)
        self.last_move.to(device=device)
        return self

    def reset(self, env_indices: torch.Tensor | None = None):
        """Resets specified game environments to their initial state."""
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
        """Performs actions in specified environments and updates their states."""
        if env_mask is None:
            env_mask = torch.ones_like(action, dtype=torch.bool)

        board_1d_view = self.board.view(self.num_envs, -1)

        values_on_board = board_1d_view[
            torch.arange(self.num_envs, device=self.device),
            action,
        ]  # (E,)

        nop = (values_on_board != 0) | (~env_mask)  # (E,)
        inc = torch.logical_not(nop).long()  # (E,)
        piece = torch.where(self.turn == 0, 1, -1)
        board_1d_view[
            torch.arange(self.num_envs, device=self.device), action
        ] = torch.where(nop, values_on_board, piece)
        self.move_count = self.move_count + inc

        # F.conv2d doesn't support LongTensor on CUDA. So we use float.
        board_one_side = (
            self.board == piece.unsqueeze(-1).unsqueeze(-1)
        ).float()
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
        """Encodes the current board state into a tensor format suitable for neural network input."""
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
        )  # (E,B,B)

        layer3 = layer3.float()

        output = torch.stack(
            [
                layer1,
                layer2,
                layer3,
            ],
            dim=1,
        )  # (E,3,B,B)
        return output

    def get_action_mask(self) -> torch.Tensor:
        """
        Priority:
        1) if current side has any move that immediately makes five, keep only those moves
        2) else if opponent has any move that would immediately make five next turn, keep only those blocking moves
        3) else keep normal legal moves
        """
        legal = (self.board == 0)

        if not self.action_pruning_enabled:
            return legal.flatten(start_dim=1)

        # Before 7 total moves placed, neither side can possibly have a "one move to five" threat.
        active = self.move_count >= 7
        if not active.any():
            return legal.flatten(start_dim=1)

        piece = torch.where(self.turn == 0, 1, -1).view(-1, 1, 1)
        mine = (self.board == piece)
        opp = (self.board == -piece)

        final_mask = legal.clone()
        remaining = active.clone()

        # Priority 1: current player can win immediately
        if self.action_pruning_self_win4 and remaining.any():
            my_mask = _compute_immediate_five_mask(
                atk=mine[remaining],
                deff=opp[remaining],
                empty=legal[remaining],
                kernel_horizontal=self.kernel_horizontal,
                kernel_vertical=self.kernel_vertical,
                kernel_diagonal=self.kernel_diagonal,
            )
            has_my = my_mask.flatten(start_dim=1).any(dim=1)
            if has_my.any():
                remaining_ids = remaining.nonzero(as_tuple=False).squeeze(-1)
                hit_ids = remaining_ids[has_my]
                final_mask[hit_ids] = my_mask[has_my]
                remaining[hit_ids] = False

        # Priority 2: block opponent's immediate win
        if self.action_pruning_block_opp_win4 and remaining.any():
            opp_mask = _compute_immediate_five_mask(
                atk=opp[remaining],
                deff=mine[remaining],
                empty=legal[remaining],
                kernel_horizontal=self.kernel_horizontal,
                kernel_vertical=self.kernel_vertical,
                kernel_diagonal=self.kernel_diagonal,
            )
            has_opp = opp_mask.flatten(start_dim=1).any(dim=1)
            if has_opp.any():
                remaining_ids = remaining.nonzero(as_tuple=False).squeeze(-1)
                hit_ids = remaining_ids[has_opp]
                final_mask[hit_ids] = opp_mask[has_opp]

        return final_mask.flatten(start_dim=1)

    def is_valid(self, action: torch.Tensor) -> torch.Tensor:
        """Checks the validity of the specified actions in each environment."""
        out_of_range = (action < 0) | (
            action >= self.board_size * self.board_size
        )
        x = action // self.board_size
        y = action % self.board_size
        values_on_board = self.board[
            torch.arange(self.num_envs, device=self.device), x, y
        ]  # (E,)
        not_empty = values_on_board != 0  # (E,)
        invalid = out_of_range | not_empty
        return ~invalid