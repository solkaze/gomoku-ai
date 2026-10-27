"""自己対局 → 学習 → (定期的に) ゲーティング を繰り返す学習ループ。

自己対局には最強モデル (model_dir/best.onnx) だけを使う。gating.interval イテレーションごとに最新モデルと
最強モデルを対戦させ、勝率が gating.threshold 以上なら最強モデルを置き換える。置き換えたときは固定の基準モデル
(gating.anchor) とも対戦させ、基準からの Elo を記録する。学習が一時的に悪くなっても、自己対局に使うモデルは
弱くならない。

途中で止まっても、再実行すれば最新のチェックポイントと自己対局データから続きを始める。
"""

import csv
import json
import subprocess
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from config import Config
from trainer import current_iteration, has_checkpoint, init, model_path, replace_file, train_iteration

BEST_MODEL = "best.onnx"
# 最強モデルがどのイテレーションか ({"iteration": int | None, "elo": float | None})。None は基準モデルそのもの
BEST_STATE = "best.json"
GATING_LOG = "gating.csv"
GATING_FIELDS = ["time", "iteration", "best", "games", "wins", "losses", "draws", "score", "promoted"]
STRENGTH_LOG = "strength.csv"
STRENGTH_FIELDS = ["time", "iteration", "games", "wins", "losses", "draws", "score", "elo"]


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
    ensure_best(config)

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
        if new_iteration % config.gating.interval == 0:
            run_gating(config, new_iteration)


def stop_requested(config: Config) -> bool:
    stop_file = config.loop.stop_file
    if stop_file.exists():
        stop_file.unlink()
        print(f"{stop_file} があったので停止します")
        return True
    return False


def best_model(config: Config) -> Path:
    return config.paths.model_dir / BEST_MODEL


def read_best(config: Config) -> dict:
    return json.loads((config.paths.log_dir / BEST_STATE).read_text())


def write_best(config: Config, iteration: int | None, elo: float | None) -> None:
    path = config.paths.log_dir / BEST_STATE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"iteration": iteration, "elo": elo}))
    tmp.replace(path)


def ensure_best(config: Config) -> None:
    """最強モデルがまだなければ基準モデルにし、今の最新モデルがそれより強ければすぐに置き換える。
    基準モデルもなければ、今の最新モデルを基準にする。"""
    if best_model(config).exists() and (config.paths.log_dir / BEST_STATE).exists():
        return
    anchor = config.gating.anchor
    iteration = current_iteration(config)
    if not anchor.exists():
        replace_file(model_path(config, iteration), anchor)
        print(f"基準モデルがないので iter {iteration} を基準にします: {anchor}")
        replace_file(anchor, best_model(config))
        write_best(config, iteration, 0.0)
        return
    replace_file(anchor, best_model(config))
    write_best(config, None, 0.0)
    print(f"最強モデルを基準モデル {anchor} にしました。iter {iteration} と比べます")
    run_gating(config, iteration)


def run_selfplay(config: Config, iteration: int) -> None:
    summary_path = config.paths.log_dir / "selfplay_summary.json"
    run_binary(config, config.loop.selfplay_bin, "--model", str(best_model(config)), "--summary", str(summary_path))
    summary = json.loads(summary_path.read_text())
    games = max(summary["games"], 1)
    with SummaryWriter(config.paths.log_dir) as writer:
        writer.add_scalar("selfplay/black_win_rate", summary["black_wins"] / games, iteration)
        writer.add_scalar("selfplay/white_win_rate", summary["white_wins"] / games, iteration)
        writer.add_scalar("selfplay/draw_rate", summary["draws"] / games, iteration)
        writer.add_scalar("selfplay/avg_moves", summary["avg_moves"], iteration)
        writer.add_scalar("selfplay/positions", summary["positions"], iteration)
        writer.add_scalar("selfplay/games_per_sec", summary["games_per_sec"], iteration)
        writer.add_scalar("selfplay/evals_per_sec", summary["evals_per_sec"], iteration)


def run_gating(config: Config, iteration: int) -> None:
    """iteration のモデルを最強モデルと対戦させ、勝率が threshold 以上なら最強モデルにする。
    置き換えたら基準モデルと対戦させ、基準からの Elo を strength.csv と TensorBoard に記録する。"""
    gc = config.gating
    candidate = model_path(config, iteration)
    best = read_best(config)
    best_name = "基準" if best["iteration"] is None else f"iter {best['iteration']}"
    summary = run_match(config, candidate, best_model(config), gc.games)
    promoted = summary["score"] >= gc.threshold
    append_row(
        config.paths.log_dir / GATING_LOG,
        GATING_FIELDS,
        {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "iteration": iteration,
            "best": "anchor" if best["iteration"] is None else best["iteration"],
            **{k: summary[k] for k in ("games", "wins", "losses", "draws", "score")},
            "promoted": int(promoted),
        },
    )
    with SummaryWriter(config.paths.log_dir) as tb:
        tb.add_scalar("gating/score_vs_best", summary["score"], iteration)
        tb.add_scalar("gating/promoted", int(promoted), iteration)
    verdict = "最強モデルを置き換えます" if promoted else f"{gc.threshold:.0%} に届かないので据え置きます"
    print(f"ゲーティング: iter {iteration} vs {best_name}: 勝率 {summary['score']:.1%} → {verdict}")
    if not promoted:
        return

    replace_file(candidate, best_model(config))
    write_best(config, iteration, None)
    strength = run_match(config, candidate, gc.anchor, gc.anchor_games)
    write_best(config, iteration, strength["elo_diff"])
    append_row(
        config.paths.log_dir / STRENGTH_LOG,
        STRENGTH_FIELDS,
        {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "iteration": iteration,
            **{k: strength[k] for k in ("games", "wins", "losses", "draws", "score")},
            "elo": strength["elo_diff"],
        },
    )
    with SummaryWriter(config.paths.log_dir) as tb:
        tb.add_scalar("strength/elo_vs_anchor", strength["elo_diff"], iteration)
    print(f"強さ: iter {iteration} vs 基準: 勝率 {strength['score']:.1%} / Elo {strength['elo_diff']:+.0f}")


def run_match(config: Config, model: Path, opponent: Path, games: int) -> dict:
    summary_path = config.paths.log_dir / "evaluate_summary.json"
    run_binary(
        config,
        config.loop.evaluate_bin,
        "--model", str(model),
        "--opponent", str(opponent),
        "--games", str(games),
        "--summary", str(summary_path),
    )  # fmt: skip
    return json.loads(summary_path.read_text())


def append_row(path: Path, fields: list[str], row: dict) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_binary(config: Config, binary: Path, *args: str) -> None:
    config.paths.log_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(binary), "--config", str(config.path), *args], check=True)
