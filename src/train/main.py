"""学習のエントリポイント。プロジェクトルートで実行する。

    uv run src/train/main.py loop     # 自己対局 → 学習 → 評価 を繰り返す (中断しても再実行で続きから)
    uv run src/train/main.py init     # ランダム初期化したモデルを作る
    uv run src/train/main.py train    # 自己対局データで 1 イテレーション学習し、ONNX を書き出す
    uv run src/train/main.py export   # 最新チェックポイントを ONNX に書き出し直す

ループを止めるときは config の loop.stop_file (既定 runs/STOP) を作る。
"""

import argparse
import sys
from pathlib import Path

import torch

from config import DEFAULT_CONFIG, load_config
from pipeline import run_loop
from trainer import export_latest, has_checkpoint, init, load, train_iteration


def main() -> None:
    # 子プロセス (selfplay / evaluate) の出力とログの順序が前後しないよう、行ごとに書き出す
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="五目並べ AI の学習")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    loop = sub.add_parser("loop", help="自己対局 → 学習 → 評価 を繰り返す")
    loop.add_argument("--iterations", type=int, help="この回数だけ学習したら終了する (省略時は止めるまで続ける)")
    init_parser = sub.add_parser("init", help="ランダム初期化したモデルを作る")
    init_parser.add_argument("--force", action="store_true", help="既存のチェックポイントを上書きする")
    sub.add_parser("train", help="自己対局データで 1 イテレーション学習する")
    sub.add_parser("export", help="最新チェックポイントを ONNX に書き出す")
    args = parser.parse_args()

    config = load_config(args.config)
    match args.command:
        case "loop":
            run_loop(config, args.iterations)
        case "init":
            if has_checkpoint(config) and not args.force:
                sys.exit("チェックポイントが既にあります。作り直す場合は --force を付けてください。")
            init(config)
        case "train":
            train_iteration(config)
        case "export":
            state = load(config, torch.device("cpu"))
            print(f"書き出しました: {export_latest(config, state.model, state.iteration)}")


if __name__ == "__main__":
    main()
