"""自己対局データの読み込みとミニバッチ生成。

Rust の自己対局が書き出す npz の形式:
    states: uint8   [N, IN_CHANNELS, 15, 15]  特徴平面 (network.py 参照)
    policy: float32 [N, 225]                  MCTS の訪問回数分布
    value:  float32 [N]                       最終結果を手番側から見た値 (勝ち 1, 負け -1, 引き分け 0)
"""

from pathlib import Path

import numpy as np
import torch

from network import BOARD_SIZE, CELLS, IN_CHANNELS


class ReplayBuffer:
    """直近 window 局面を GPU 上に保持し、8 対称の拡張をかけてサンプリングする。"""

    def __init__(self, states: np.ndarray, policy: np.ndarray, value: np.ndarray, device: torch.device) -> None:
        self.states = torch.from_numpy(states).to(device)
        self.policy = torch.from_numpy(policy).to(device)
        self.value = torch.from_numpy(value).to(device)

    def __len__(self) -> int:
        return len(self.value)

    @classmethod
    def load(
        cls, selfplay_dir: Path, window: int, device: torch.device, policy_dtype: type = np.float32
    ) -> "ReplayBuffer":
        """直近 window 局面を読む。"""
        files = newest_files(selfplay_dir, window)
        if not files:
            raise FileNotFoundError(f"自己対局データがありません: {selfplay_dir}")
        return cls.from_files(files, device, window, policy_dtype)

    @classmethod
    def from_files(
        cls, files: list[Path], device: torch.device, window: int | None = None, policy_dtype: type = np.float32
    ) -> "ReplayBuffer":
        """新しい順に並んだ files を読み、(window があれば) 末尾 window 局面だけ残す。
        policy_dtype を float16 にすると、大量の局面を読むときのメモリを減らせる。"""
        chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for path in files:
            with np.load(path) as data:
                states, policy, value = data["states"], data["policy"], data["value"]
            _validate(path, states, policy, value)
            chunks.append((states, policy.astype(policy_dtype, copy=False), value))
        # 新しい順に読んだので古い順に戻す
        chunks.reverse()
        states, policy, value = (np.concatenate(xs)[-window if window else 0 :] for xs in zip(*chunks))
        return cls(states, policy, value, device)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.randint(len(self), (batch_size,), device=self.value.device)
        return self.batch(idx)

    def batch(
        self, idx: torch.Tensor, augmented: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """idx の局面を学習用のテンソルにする。augmented なら 8 対称の拡張をかける。"""
        states = self.states[idx].float()
        policy = self.policy[idx].float()
        if augmented:
            states, policy = augment(states, policy.view(-1, 1, BOARD_SIZE, BOARD_SIZE))
        return states, policy.reshape(-1, CELLS), self.value[idx]


def newest_files(selfplay_dir: Path, positions: int) -> list[Path]:
    """自己対局ファイルを新しい順に、合計が positions 局面に達するまで返す。"""
    files = sorted(selfplay_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    picked, total = [], 0
    for path in files:
        with np.load(path) as data:
            total += len(data["value"])
        picked.append(path)
        if total >= positions:
            break
    return picked


def augment(states: torch.Tensor, policy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """盤面の 8 対称 (回転 4 × 反転 2) をサンプルごとにランダムにかける。

    連珠のルール (禁じ手を含む) は盤面の対称変換で不変なので、特徴平面ごと変換してよい。
    """
    sym = torch.randint(8, (len(states),), device=states.device)
    states, policy = states.clone(), policy.clone()
    for s in range(1, 8):
        mask = sym == s
        if mask.any():
            states[mask] = transform(states[mask], s)
            policy[mask] = transform(policy[mask], s)
    return states, policy


def transform(x: torch.Tensor, sym: int) -> torch.Tensor:
    """[B, C, H, W] に対称変換 sym (0〜7) をかける。"""
    x = torch.rot90(x, sym % 4, dims=(2, 3))
    return x.flip(3) if sym >= 4 else x


def _validate(path: Path, states: np.ndarray, policy: np.ndarray, value: np.ndarray) -> None:
    n = len(value)
    expected = {
        "states": (states, (n, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE), np.uint8),
        "policy": (policy, (n, CELLS), np.float32),
        "value": (value, (n,), np.float32),
    }
    for name, (array, shape, dtype) in expected.items():
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"{path}: {name} は {shape} {dtype} のはずが {array.shape} {array.dtype}")
