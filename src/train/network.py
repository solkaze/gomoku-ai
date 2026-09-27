"""方策・価値ネットワーク (AlphaZero 型 ResNet)。

入力は Rust 側で作る特徴平面 [B, IN_CHANNELS, 15, 15]:
    0: 手番側の石
    1: 相手の石
    2: 手番が黒なら全て 1
    3: 黒の禁じ手の点
出力は方策ロジット [B, 225] と、手番側から見た価値 [B] (tanh, -1〜1)。
"""

import torch
import torch.nn.functional as F
from torch import nn

from config import ModelConfig

BOARD_SIZE = 15
CELLS = BOARD_SIZE * BOARD_SIZE
IN_CHANNELS = 4


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, channels: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(*(ResBlock(channels) for _ in range(blocks)))
        # 方策は全結合を使わず 1x1 畳み込みで各点のロジットを出す
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * CELLS, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    @classmethod
    def from_config(cls, config: ModelConfig) -> "PolicyValueNet":
        return cls(config.channels, config.blocks)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self.stem(x))
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def loss_fn(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    value_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(合計損失, 方策の交差エントロピー, 価値の二乗誤差) を返す。"""
    policy_loss = -(target_policy * F.log_softmax(policy_logits.float(), dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(value.float(), target_value)
    return policy_loss + value_weight * value_loss, policy_loss, value_loss
