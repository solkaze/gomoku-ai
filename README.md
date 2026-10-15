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
uv run src/train/main.py loop           # 自己対局 → 学習 → ゲーティング を繰り返す
```

- 途中で止まっても、再実行すれば最新のチェックポイントから続ける。
- `touch runs/STOP` で、今の段階 (自己対局 or 学習) が終わったところで停止する。
- 進捗は `uv run tensorboard --logdir runs/logs` で見る (損失・学習率・自己対局の勝率・Elo)。
- 自己対局には最強モデル `runs/models/best.onnx` だけを使う (ゲーティング)。`gating.interval` イテレーションごとに
  最新モデルと `gating.games` 局打たせ、勝率が `gating.threshold` 以上なら置き換える。結果は `runs/logs/gating.csv`。
  学習が一時的に悪くなっても、自己対局に使うモデルは弱くならない。
- 最強モデルが入れ替わるたびに固定の基準モデル `gating.anchor` と対戦させ、基準からの Elo を
  `runs/logs/strength.csv` に記録する。強くなっているかはこれで見る。
- 自己対局は playout cap randomization (KataGo) を使う。`selfplay.full_search_prob` の割合の手だけ
  `simulations` 回探索して学習データにし、残りは `fast_simulations` 回の速い探索で打つ。

### ネットワークを大きくする

強化学習を最初からやり直さず、今ある自己対局データで大きいネットワークを教師あり学習してから引き継ぐ。

```sh
uv run src/train/main.py distill --channels 192 --blocks 15 --out runs/distill/15b192c.pt
uv run src/train/main.py adopt runs/distill/15b192c.pt   # 今のチェックポイントは backup_iter_XXXX.pt に残る
rm runs/models/best.onnx                                  # 次のループ開始時に基準モデルと比べて選び直させる
```

`distill` は直近 `--positions` 局面 (既定 500 万) を `--epochs` 周学習し、検証損失を今のモデルと並べて表示する。
`adopt` 後の最初のループでは、最強モデルを基準モデルに戻したうえで新しいモデルと対戦させ、
勝ち越したときだけ自己対局に使う。

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
./target/release/gomoku --free                  # 自由対局 (AI なし。モデルを読み込まない)
./target/release/gomoku --allow overline        # 長連を禁じ手から外す (double-three, double-four も。カンマ区切りで複数)
```

起動するとモデルの読み込みの進み具合 (CUDA ライブラリ → ONNX Runtime → モデル → 試し推論) を表示し、終わると先手・後手の選択に移る (ここで 3 つ目の「自由対局」も選べる)。

- 盤は色付きで描き、黒石・白石を大きな丸で表す。端末が横 103 x 縦 33 文字以上なら大きい盤、それ未満なら小さい盤 (横 77 x 縦 18 文字以上が必要) になる。
- 矢印キーかマウスでカーソルを動かし、Enter・Space・クリックで打つ。黒の初手 (天元) は自動で打つ。
- 黒 (先手) には三三・四四・長連の禁じ手を人・AI のどちらにも適用する。禁じ手の点にも石は置けるが、置いた時点で黒の負けになる。人が黒のときは禁じ手の点に赤い × を表示する。AI は他に打てる点がある限り禁じ手を打たない。
- 直前の手には赤い印が付き、勝ちが決まった石の並びは金色で示す。
- AI は手を打つたびに、勝率の見込みと探索の候補手 (訪問回数の割合) を表示する。
- キー: `u` (待った。自分の直前の手まで戻す)、`h` (ヒント。今の局面での AI の候補手を盤上に青く表示)、`n` (新しい対局)、`q` / `Esc` / `Ctrl-C` (終了)。
- 黒の禁じ手は種類ごとに無効にできる (ハンデなど)。`--allow overline,double-three` のように外す手を指定するか、起動後の選択画面で `4`〜`6` キー (三三・四四・長連) かクリックで切り替える。長連を外すと黒も六以上の連で勝ちになる。学習 (自己対局・評価) は常に全ての禁じ手を有効にして行うので、AI は無効にした禁じ手に関わる形 (黒の三三や長連での勝ちなど) を学習しておらず、その部分は探索だけが頼りになる。
- 自由対局 (`--free`、または起動後の選択) では AI を使わず、黒と白の両方を自分で交互に打つ。ルール (初手は天元、黒の禁じ手) は同じ。`u` は 1 手戻し、ヒントは使えない。
- 情報欄に、対局開始からの時間と両者の累計の考慮時間、今の手番の考慮時間、人が直前に打った手にかかった時間を表示する (AI の手の所要時間は従来どおり)。
- 終了すると棋譜と、対局時間・各手にかかった秒数を表示する。座標は `x y` (x は左から、y は上から、どちらも 0〜14) で、天元は `7 7`。
- `--plain` を付けるか、入出力が端末でない (パイプなど) 場合は、色を使わず座標を `7 7` のように入力する行単位の表示になる (`X` が黒、`O` が白、`[ ]` が直前の手、`!` が黒の禁じ手 (打つと負け)。コマンドは `undo`、`hint`、`help`、`quit`)。

## 主な設定 (`config/config.toml`)

| 項目 | 意味 |
|---|---|
| `train.samples_per_position` | 新しい局面 1 つあたりの学習サンプル数。学習ステップ数はここから決まる |
| `train.lr_schedule` | 累計ステップ数ごとの学習率 |
| `train.replay_window` / `min_replay` | 学習に使う直近の局面数 / 学習を始める局面数 |
| `selfplay.games` / `simulations` | 1 イテレーションの自己対局数 / 1 手あたりの探索回数 |
| `selfplay.full_search_prob` / `fast_simulations` | playout cap randomization: 通常の探索をする手の割合 (1.0 で無効) / それ以外の手の探索回数 |
| `selfplay.opening_moves` / `evaluate.opening_moves` | 開局でランダムに打つ手数の上限 (0 で無効)。評価では同じ開局を先後入れ替えて打つ |
| `evaluate.simulations` | 対戦 (ゲーティング・基準モデルとの対戦) での 1 手あたりの探索回数 |
| `gating.interval` / `games` / `threshold` | 最強モデルと比べる間隔 / 対局数 / 置き換える勝率 |
| `gating.anchor` / `anchor_games` | 強さの基準にする固定のモデル / それとの対局数 |
| `loop.keep_last` / `keep_every` | 残すチェックポイント (直近 N 個と N イテレーションごと) |
| `inference.threads` / `max_batch` | GPU 推論のスレッド数とバッチサイズ |

## テスト

```sh
cargo test --lib
uv run pytest src/train
```
