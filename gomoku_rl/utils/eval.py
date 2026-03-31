import torch
from tensordict.nn import InteractionType, set_interaction_type

from gomoku_rl.env import GomokuEnv
from gomoku_rl.utils.policy import _policy_t


class MatchStats:
    def __init__(
        self,
        black_win_rate: float,
        white_win_rate: float,
        draw_rate: float,
    ) -> None:
        self.black_win_rate = float(black_win_rate)
        self.white_win_rate = float(white_win_rate)
        self.draw_rate = float(draw_rate)

    @property
    def black_score(self) -> float:
        return self.black_win_rate + 0.5 * self.draw_rate

    @property
    def white_score(self) -> float:
        return self.white_win_rate + 0.5 * self.draw_rate


def eval_win_rate(
    env: GomokuEnv,
    player_black: _policy_t,
    player_white: _policy_t,
    n: int = 1,
) -> float:
    return eval_match(
        env=env,
        player_black=player_black,
        player_white=player_white,
        n=n,
    ).black_win_rate


def eval_match(
    env: GomokuEnv,
    player_black: _policy_t,
    player_white: _policy_t,
    n: int = 1,
) -> MatchStats:
    tmp = [_eval_match_once(env, player_black, player_white) for _ in range(n)]

    black_win_rate = sum(item.black_win_rate for item in tmp) / len(tmp)
    white_win_rate = sum(item.white_win_rate for item in tmp) / len(tmp)
    draw_rate = sum(item.draw_rate for item in tmp) / len(tmp)

    return MatchStats(
        black_win_rate=black_win_rate,
        white_win_rate=white_win_rate,
        draw_rate=draw_rate,
    )


@set_interaction_type(type=InteractionType.RANDOM)
@torch.no_grad()
def _eval_match_once(
    env: GomokuEnv,
    player_black: _policy_t,
    player_white: _policy_t,
) -> MatchStats:
    board_size = env.board_size
    tensordict = env.reset()

    if hasattr(player_black, "eval"):
        player_black.eval()
    if hasattr(player_white, "eval"):
        player_white.eval()

    episode_done = torch.zeros(
        env.num_envs,
        device=tensordict.device,
        dtype=torch.bool,
    )
    interested_tensordict = []

    for i in range(board_size * board_size + 1):
        if i % 2 == 0:
            tensordict = player_black(tensordict)
        else:
            tensordict = player_white(tensordict)

        tensordict = env.step_and_maybe_reset(tensordict)
        done = tensordict.get("done")

        index: torch.Tensor = done & ~episode_done
        episode_done = episode_done | done
        interested_tensordict.extend(tensordict["stats"][index].unbind(0))

        if episode_done.all().item():
            break

    env.reset()

    if hasattr(player_black, "train"):
        player_black.train()
    if hasattr(player_white, "train"):
        player_white.train()

    interested_tensordict = torch.stack(interested_tensordict, dim=0)
    black_win_rate = interested_tensordict["black_win"].float().mean().item()
    white_win_rate = interested_tensordict["white_win"].float().mean().item()
    draw_rate = 1.0 - black_win_rate - white_win_rate

    return MatchStats(
        black_win_rate=black_win_rate,
        white_win_rate=white_win_rate,
        draw_rate=draw_rate,
    )


def get_payoff_matrix(
    env: GomokuEnv,
    row_policies: list[_policy_t],
    col_policies: list[_policy_t],
    n: int = 1,
):
    payoff = torch.zeros(len(row_policies), len(col_policies))
    for i, row_p in enumerate(row_policies):
        for j, col_p in enumerate(col_policies):
            payoff[i, j] = eval_win_rate(
                env,
                player_black=row_p,
                player_white=col_p,
                n=n,
            )
    return payoff