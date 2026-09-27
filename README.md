# gomoku-ai

連珠ルール (黒のみ三三・四四・長連禁止、黒の初手は天元固定) の 15×15 五目並べ AI。
AlphaZero 方式 (MCTS + 方策・価値ネットワーク) で自己対局から学習する。

| 部分 | 言語 | 場所 |
|---|---|---|
| ルール・MCTS・自己対局・評価・対戦 | Rust (推論は ort / CUDA) | `src/engine`, `src/selfplay`, `src/evaluate`, `src/run` |
| ネットワークの学習・ONNX 書き出し・学習ループ | Python (PyTorch) | `src/train` |

設定はすべて `config/config.toml`。成果物は `runs/` 以下に出る。

## 学習

```sh
uv sync
cargo build --release
uv run src/train/main.py loop           # 自己対局 → 学習 → 評価 を繰り返す
```

- 途中で止まっても、再実行すれば最新のチェックポイントから続ける。
- `touch runs/STOP` で、今の段階 (自己対局 or 学習) が終わったところで停止する。
- 進捗は `uv run tensorboard --logdir runs/logs` で見る (損失・学習率・自己対局の勝率・Elo)。
- 強さの推移は `runs/logs/evaluation.csv` にも記録される。`evaluate.interval` イテレーションごとに、
  前回の評価時点のモデルと対戦した勝率と、そこから積み上げた累計 Elo。

### サーバーで回す (Docker)

NVIDIA ドライバと NVIDIA Container Toolkit が入ったマシンで:

```sh
mkdir -p runs                           # 先に作る (ないと Docker が root 所有で作ってしまう)
docker compose up -d --build train      # 学習開始 (異常終了時は自動で再起動し続きから)
docker compose logs -f train
docker compose up -d tensorboard        # http://<サーバー>:6006
touch runs/STOP                         # 停止
```

コンテナはホストのユーザー (既定 UID/GID 1000) で動くので、`runs/` のファイルは sudo なしで扱える。
UID/GID が 1000 でなければ `.env` に `UID=...` / `GID=...` を書く。
事前に `docker run --rm --gpus all ubuntu nvidia-smi` で GPU が見えるか確認しておく (NVIDIA Container Toolkit が必要)。

イメージには `uv.lock` / `Cargo.lock` / `rust-toolchain.toml` どおりの依存とビルド済みバイナリが入る。
CUDA / cuDNN は torch の wheel に同梱のもの (`.venv` 内) を Rust 側でも使うので、ホストに CUDA Toolkit は不要。
ビルド時に crates.io・PyPI・onnxruntime のバイナリ (cdn.pyke.io) へのネットワーク接続が必要。

## 主な設定 (`config/config.toml`)

| 項目 | 意味 |
|---|---|
| `train.samples_per_position` | 新しい局面 1 つあたりの学習サンプル数。学習ステップ数はここから決まる |
| `train.lr_schedule` | 累計ステップ数ごとの学習率 |
| `train.replay_window` / `min_replay` | 学習に使う直近の局面数 / 学習を始める局面数 |
| `selfplay.games` / `simulations` | 1 イテレーションの自己対局数 / 1 手あたりの探索回数 |
| `evaluate.interval` / `games` | 強さを測る間隔 / 対局数 |
| `loop.keep_last` / `keep_every` | 残すチェックポイント (直近 N 個と N イテレーションごと) |
| `inference.threads` / `max_batch` | GPU 推論のスレッド数とバッチサイズ |

## テスト

```sh
cargo test --lib
uv run pytest src/train
```
