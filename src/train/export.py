"""学習済みネットワークを Rust (ort) 用の ONNX に書き出す。

入力 "input" [batch, IN_CHANNELS, 15, 15] float32
出力 "policy" [batch, 225] (ロジット), "value" [batch] (tanh)
fp16 で書き出しても入出力は float32 のまま (内部の計算だけ float16)。
"""

import copy
import os
from pathlib import Path

import torch

from torch import nn

from network import BOARD_SIZE, IN_CHANNELS, PolicyValueNet


class HalfPrecision(nn.Module):
    """入出力は float32 のまま、内部を float16 で計算する。"""

    def __init__(self, model: PolicyValueNet) -> None:
        super().__init__()
        self.model = model.half()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy, value = self.model(x.half())
        return policy.float(), value.float()


def export_onnx(model: PolicyValueNet, path: Path, fp16: bool = False) -> None:
    """推論モードのコピーを書き出す。書き込み途中のファイルを Rust が読まないよう最後に置き換える。"""
    model = copy.deepcopy(model).cpu().float().eval()
    if fp16:
        model = HalfPrecision(model).eval()
    dummy = torch.zeros(2, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.onnx")
    torch.onnx.export(
        model,
        (dummy,),
        tmp,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
        dynamo=True,
        external_data=False,
        verbose=False,
    )
    os.replace(tmp, path)
