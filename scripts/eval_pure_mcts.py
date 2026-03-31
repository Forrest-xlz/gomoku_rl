import hydra
from omegaconf import DictConfig, OmegaConf

from gomoku_rl import CONFIG_PATH
from gomoku_rl.env import GomokuEnv
from gomoku_rl.mcts import PureMCTSPlayer
from gomoku_rl.policy import get_pretrained_policy
from gomoku_rl.utils.eval import eval_match


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="eval_pure_mcts")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    env = GomokuEnv(
        num_envs=cfg.num_envs,
        board_size=cfg.board_size,
        device=cfg.device,
    )

    black_player = get_pretrained_policy(
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        checkpoint_path=cfg.black_checkpoint,
        device=env.device,
    )

    white_player = get_pretrained_policy(
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=env.action_spec,
        observation_spec=env.observation_spec,
        checkpoint_path=cfg.white_checkpoint,
        device=env.device,
    )

    pure_mcts = PureMCTSPlayer(
        board_size=cfg.board_size,
        num_simulations=cfg.num_simulations,
        c_puct=cfg.get("c_puct", 5.0),
        rollout_limit=cfg.get("rollout_limit", None),
        seed=cfg.get("seed", 0),
    )

    black_stats = eval_match(
        env=env,
        player_black=black_player,
        player_white=pure_mcts,
        n=cfg.num_matches,
    )
    white_stats = eval_match(
        env=env,
        player_black=pure_mcts,
        player_white=white_player,
        n=cfg.num_matches,
    )

    print(
        f"player as black vs pure mcts({cfg.num_simulations}): {black_stats.black_win_rate:.4f}"
    )
    print(
        f"player as white vs pure mcts({cfg.num_simulations}): {white_stats.white_win_rate:.4f}"
    )


if __name__ == "__main__":
    main()
