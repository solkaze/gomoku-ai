"""今ある自己対局データで、新しい (大きくした) ネットワークを教師あり学習する。

ネットワークを大きくするとき、強化学習を最初からやり直さずに今のモデルの強さまで育てるために使う。
直近 positions 局面を epochs 周学習する。50 ファイルに 1 つは検証用に取り分け、学習には使わない。

    uv run src/train/main.py distill --channels 192 --blocks 15 --out runs/distill/15b192c.pt
    uv run src/train/main.py adopt runs/distill/15b192c.pt     # 学習ループの最新モデルにする
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config, ModelConfig
from export import export_onnx
from network import PolicyValueNet, loss_fn
from replay import ReplayBuffer, newest_files
from trainer import LATEST_CHECKPOINT, has_checkpoint, make_optimizer, model_channels

VALIDATION_EVERY = 50
# 学習率: warmup ステップで lr まで上げ、そこから cosine で lr * MIN_LR_RATIO まで下げる
WARMUP_STEPS = 1000
MIN_LR_RATIO = 0.01


@dataclass(frozen=True)
class DistillArgs:
    model: ModelConfig
    positions: int
    epochs: float
    lr: float
    out: Path


def lr_at_step(step: int, steps: int, lr: float) -> float:
    warmup = min(WARMUP_STEPS, max(steps // 10, 1))
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(steps - warmup, 1)
    return lr * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def validate(model: PolicyValueNet, buffer: ReplayBuffer, config: Config) -> tuple[float, float]:
    """検証データ全体の (方策の交差エントロピー, 価値の二乗誤差)。"""
    device = buffer.value.device
    use_amp = config.train.amp and device.type == "cuda"
    model.eval()
    policy_sum = value_sum = 0.0
    for start in range(0, len(buffer), 4096):
        idx = torch.arange(start, min(start + 4096, len(buffer)), device=device)
        states, target_policy, target_value = buffer.batch(idx, augmented=False)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits, value = model(states)
        _, policy_loss, value_loss = loss_fn(logits, value, target_policy, target_value, 1.0)
        policy_sum += policy_loss.item() * len(idx)
        value_sum += value_loss.item() * len(idx)
    model.train()
    return policy_sum / len(buffer), value_sum / len(buffer)


def current_model(config: Config, device: torch.device) -> PolicyValueNet | None:
    """比較用に、学習ループの最新モデルを読む (なければ None)。"""
    if not has_checkpoint(config):
        return None
    data = torch.load(config.paths.checkpoint_dir / LATEST_CHECKPOINT, map_location=device, weights_only=True)
    model = PolicyValueNet.from_config(ModelConfig(**data["model_config"])).to(device)
    model.load_state_dict(data["model"])
    return model


def distill(config: Config, args: DistillArgs) -> None:
    tc = config.train
    device = torch.device(tc.device)
    files = newest_files(config.paths.selfplay_dir, args.positions)
    if not files:
        raise SystemExit(f"自己対局データがありません: {config.paths.selfplay_dir}")
    val_files = files[::VALIDATION_EVERY]
    train_files = [p for i, p in enumerate(files) if i % VALIDATION_EVERY]
    print(f"{len(files)} ファイルを読み込みます (うち検証用 {len(val_files)})")
    # 方策は float16 で持ち、メモリを半分にする
    train = ReplayBuffer.from_files(train_files, device, policy_dtype=np.float16)
    val = ReplayBuffer.from_files(val_files, device, policy_dtype=np.float16)
    steps = math.ceil(len(train) * args.epochs / tc.batch_size)
    print(f"学習 {len(train)} 局面 / 検証 {len(val)} 局面 / {args.model} を {steps} ステップ学習します")

    writer = SummaryWriter(config.paths.log_dir / "distill" / args.out.stem)
    reference = current_model(config, device)
    if reference is not None:
        ref_policy, ref_value = validate(reference, val, config)
        print(f"比較: 今のモデルの検証損失 policy {ref_policy:.4f} / value {ref_value:.4f}")
        writer.add_scalar("val/reference_policy", ref_policy, 0)
        writer.add_scalar("val/reference_value", ref_value, 0)
        del reference

    torch.manual_seed(0)
    model = PolicyValueNet.from_config(args.model).to(device)
    optimizer = make_optimizer(model, config)
    use_amp = tc.amp and device.type == "cuda"
    eval_every = max(steps // 20, 1)
    model.train()
    for step in tqdm(range(steps), desc="distill", mininterval=10):
        lr = lr_at_step(step, steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        states, target_policy, target_value = train.sample(tc.batch_size)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits, value = model(states)
        loss, policy_loss, value_loss = loss_fn(logits, value, target_policy, target_value, tc.value_loss_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            writer.add_scalar("train/policy", policy_loss.item(), step)
            writer.add_scalar("train/value", value_loss.item(), step)
            writer.add_scalar("train/lr", lr, step)
        if (step + 1) % eval_every == 0 or step == steps - 1:
            val_policy, val_value = validate(model, val, config)
            writer.add_scalar("val/policy", val_policy, step + 1)
            writer.add_scalar("val/value", val_value, step + 1)
            tqdm.write(f"step {step + 1}/{steps}: 検証損失 policy {val_policy:.4f} / value {val_value:.4f} (lr {lr:.2e})")
    writer.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": {"channels": model_channels(model), "blocks": len(model.trunk)},
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        args.out,
    )
    onnx = args.out.with_suffix(".onnx")
    export_onnx(model, onnx, fp16=tc.export_fp16)
    print(f"保存しました: {args.out} / {onnx}")
