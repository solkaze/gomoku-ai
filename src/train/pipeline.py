"""自己対局 → 学習 → (定期的に) 強さの評価 を繰り返す学習ループ。

途中で止まっても、再実行すれば最新のチェックポイントと自己対局データから続きを始める。
"""

import csv
import json
import subprocess
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from config import Config
from trainer import current_iteration, has_checkpoint, init, model_path, train_iteration

EVAL_LOG = "evaluation.csv"
EVAL_FIELDS = ["time", "iteration", "opponent", "games", "wins", "losses", "draws", "score", "elo_diff", "elo"]


def run_loop(config: Config, max_iterations: int | None) -> None:
    for binary in (config.loop.selfplay_bin, config.loop.evaluate_bin):
        if not binary.exists():
            raise SystemExit(f"{binary} がありません。先に `cargo build --release` を実行してください。")
    for directory in (config.paths.selfplay_dir, config.paths.model_dir, config.paths.checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
        # 前回中断したときの書きかけのファイル
        for tmp in directory.glob("*.tmp"):
            tmp.unlink()
    if not has_checkpoint(config):
        print("チェックポイントがないので初期モデルを作ります")
        init(config)

    trained = 0
    while max_iterations is None or trained < max_iterations:
        iteration = current_iteration(config)
        if stop_requested(config):
            return
        run_selfplay(config, iteration)
        if stop_requested(config):
            return
        new_iteration = train_iteration(config)
        if new_iteration is None:
            continue
        trained += 1
        interval = config.evaluate.interval
        if new_iteration % interval == 0 and new_iteration >= interval:
            run_evaluation(config, new_iteration, new_iteration - interval)


def stop_requested(config: Config) -> bool:
    stop_file = config.loop.stop_file
    if stop_file.exists():
        stop_file.unlink()
        print(f"{stop_file} があったので停止します")
        return True
    return False


def run_selfplay(config: Config, iteration: int) -> None:
    summary_path = config.paths.log_dir / "selfplay_summary.json"
    run_binary(config, config.loop.selfplay_bin, "--summary", str(summary_path))
    summary = json.loads(summary_path.read_text())
    games = max(summary["games"], 1)
    with SummaryWriter(config.paths.log_dir) as writer:
        writer.add_scalar("selfplay/black_win_rate", summary["black_wins"] / games, iteration)
        writer.add_scalar("selfplay/white_win_rate", summary["white_wins"] / games, iteration)
        writer.add_scalar("selfplay/draw_rate", summary["draws"] / games, iteration)
        writer.add_scalar("selfplay/avg_moves", summary["avg_moves"], iteration)
        writer.add_scalar("selfplay/games_per_sec", summary["games_per_sec"], iteration)
        writer.add_scalar("selfplay/evals_per_sec", summary["evals_per_sec"], iteration)


def run_evaluation(config: Config, iteration: int, opponent: int) -> None:
    """iteration のモデルを opponent のモデルと対戦させ、累計 Elo を evaluation.csv と TensorBoard に記録する。"""
    model, opponent_model = model_path(config, iteration), model_path(config, opponent)
    if not opponent_model.exists():
        print(f"評価の相手 {opponent_model} がないので評価を飛ばします")
        return
    summary_path = config.paths.log_dir / "evaluate_summary.json"
    run_binary(
        config,
        config.loop.evaluate_bin,
        "--model", str(model),
        "--opponent", str(opponent_model),
        "--summary", str(summary_path),
    )  # fmt: skip
    summary = json.loads(summary_path.read_text())

    log_path = config.paths.log_dir / EVAL_LOG
    previous_elo = 0.0
    if log_path.exists():
        with open(log_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            previous_elo = float(rows[-1]["elo"])
    elo = previous_elo + summary["elo_diff"]
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "iteration": iteration,
        "opponent": opponent,
        **{k: summary[k] for k in ("games", "wins", "losses", "draws", "score", "elo_diff")},
        "elo": elo,
    }
    write_header = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    with SummaryWriter(config.paths.log_dir) as tb:
        tb.add_scalar("eval/score_vs_previous", summary["score"], iteration)
        tb.add_scalar("eval/elo_diff", summary["elo_diff"], iteration)
        tb.add_scalar("eval/elo", elo, iteration)
    print(f"評価: iter {iteration} vs iter {opponent}: 勝率 {summary['score']:.1%} / 累計 Elo {elo:+.0f}")


def run_binary(config: Config, binary: Path, *args: str) -> None:
    config.paths.log_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(binary), "--config", str(config.path), *args], check=True)
