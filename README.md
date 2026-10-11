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
CUDA / cuDNN は torch の wheel に同梱のもの、onnxruntime は onnxruntime-gpu の公式 wheel (`.venv` 内) を
Rust 側でも実行時に読み込むので、ホストに CUDA Toolkit は不要 (RTX 50 系を含む sm_75〜sm_120 に対応)。
ビルド時に crates.io・PyPI へのネットワーク接続が必要。

## AI と対戦する

```sh
cargo build --release
./target/release/gomoku                         # runs/models/latest.onnx と対戦。先手・後手は起動後に選ぶ
./target/release/gomoku --color white           # 尋ねずに人が白 (後手) で始める
./target/release/gomoku --model runs/models/iter_0300.onnx --simulations 1600
./target/release/gomoku --cpu                   # GPU を学習に使っているときなど
./target/release/gomoku --plain                 # 色やカーソルを使わない行単位の表示
```

起動するとモデルの読み込みの進み具合 (CUDA ライブラリ → ONNX Runtime → モデル → 試し推論) を表示し、終わると先手・後手の選択に移る。

- 盤は色付きで描き、黒石・白石を大きな丸で表す。端末が横 103 x 縦 33 文字以上なら大きい盤、それ未満なら小さい盤 (横 77 x 縦 18 文字以上が必要) になる。
- 矢印キーかマウスでカーソルを動かし、Enter・Space・クリックで打つ。黒の初手 (天元) は自動で打つ。
- 黒 (先手) には三三・四四・長連の禁じ手を人・AI のどちらにも適用する。禁じ手の点にも石は置けるが、置いた時点で黒の負けになる。人が黒のときは禁じ手の点に赤い × を表示する。AI は他に打てる点がある限り禁じ手を打たない。
- 直前の手には赤い印が付き、勝ちが決まった石の並びは金色で示す。
- AI は手を打つたびに、勝率の見込みと探索の候補手 (訪問回数の割合) を表示する。
- キー: `u` (待った。自分の直前の手まで戻す)、`h` (ヒント。今の局面での AI の候補手を盤上に青く表示)、`n` (新しい対局)、`q` / `Esc` / `Ctrl-C` (終了)。
- 終了すると棋譜を表示する。座標は `x y` (x は左から、y は上から、どちらも 0〜14) で、天元は `7 7`。
- `--plain` を付けるか、入出力が端末でない (パイプなど) 場合は、色を使わず座標を `7 7` のように入力する行単位の表示になる (`X` が黒、`O` が白、`[ ]` が直前の手、`!` が黒の禁じ手 (打つと負け)。コマンドは `undo`、`hint`、`help`、`quit`)。

## 主な設定 (`config/config.toml`)

| 項目 | 意味 |
|---|---|
| `train.samples_per_position` | 新しい局面 1 つあたりの学習サンプル数。学習ステップ数はここから決まる |
| `train.lr_schedule` | 累計ステップ数ごとの学習率 |
| `train.replay_window` / `min_replay` | 学習に使う直近の局面数 / 学習を始める局面数 |
| `selfplay.games` / `simulations` | 1 イテレーションの自己対局数 / 1 手あたりの探索回数 |
| `selfplay.opening_moves` / `evaluate.opening_moves` | 開局でランダムに打つ手数の上限 (0 で無効)。評価では同じ開局を先後入れ替えて打つ |
| `evaluate.interval` / `games` | 強さを測る間隔 / 対局数 |
| `loop.keep_last` / `keep_every` | 残すチェックポイント (直近 N 個と N イテレーションごと) |
| `inference.threads` / `max_batch` | GPU 推論のスレッド数とバッチサイズ |

## テスト

```sh
cargo test --lib
uv run pytest src/train
```
