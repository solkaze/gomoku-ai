"""チェックポイントの保存・読み込みと 1 イテレーション分の学習。"""

import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config, ModelConfig
from export import export_onnx
from network import PolicyValueNet, loss_fn
from replay import ReplayBuffer

LATEST_CHECKPOINT = "latest.pt"
LATEST_MODEL = "latest.onnx"
ITER_PATTERN = re.compile(r"iter_(\d+)")


@dataclass
class TrainState:
    model: PolicyValueNet
    optimizer: torch.optim.Optimizer
    iteration: int
    # これまでの累計学習ステップ数 (学習率スケジュールに使う)
    step: int
    # 学習に使った自己対局ファイルのうち最も新しい更新時刻。これより新しいファイルが「新しいデータ」
    trained_until: float


def iter_name(iteration: int) -> str:
    return f"iter_{iteration:04d}"


def model_path(config: Config, iteration: int) -> Path:
    return config.paths.model_dir / f"{iter_name(iteration)}.onnx"


def lr_at(schedule: list[tuple[int, float]], step: int) -> float:
    lr = schedule[0][1]
    for start, value in schedule:
        if step >= start:
            lr = value
    return lr


def make_optimizer(model: PolicyValueNet, config: Config) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=lr_at(config.train.lr_schedule, 0), weight_decay=config.train.weight_decay
    )


def replace_file(src: Path, dst: Path) -> None:
    """dst を読んでいるプロセスが途中の内容を見ないよう、コピーしてから置き換える。"""
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def model_channels(model: PolicyValueNet) -> int:
    return model.stem[0].out_channels


def export_latest(config: Config, model: PolicyValueNet, iteration: int) -> Path:
    onnx = model_path(config, iteration)
    export_onnx(model, onnx, fp16=config.train.export_fp16)
    replace_file(onnx, config.paths.model_dir / LATEST_MODEL)
    return onnx


def save(config: Config, state: TrainState) -> None:
    ckpt_dir = config.paths.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{iter_name(state.iteration)}.pt"
    tmp = ckpt.with_name(ckpt.name + ".tmp")
    torch.save(
        {
            "iteration": state.iteration,
            "step": state.step,
            "trained_until": state.trained_until,
            "model_config": {"channels": model_channels(state.model), "blocks": len(state.model.trunk)},
            "model": state.model.state_dict(),
            "optimizer": state.optimizer.state_dict(),
        },
        tmp,
    )
    os.replace(tmp, ckpt)
    replace_file(ckpt, ckpt_dir / LATEST_CHECKPOINT)
    onnx = export_latest(config, state.model, state.iteration)
    prune(config, state.iteration)
    print(f"保存しました: {ckpt} / {onnx}")


def has_checkpoint(config: Config) -> bool:
    return (config.paths.checkpoint_dir / LATEST_CHECKPOINT).exists()


def load(config: Config, device: torch.device) -> TrainState:
    path = config.paths.checkpoint_dir / LATEST_CHECKPOINT
    if not path.exists():
        sys.exit(f"チェックポイントがありません: {path}\n先に `init` を実行してください。")
    data = torch.load(path, map_location=device, weights_only=True)
    # 構造はチェックポイントに合わせる (設定ファイルを後から変えても読めるように)
    model_config = ModelConfig(**data["model_config"])
    if model_config != config.model:
        print(f"注意: 設定 {config.model} ではなくチェックポイントの {model_config} を使います")
    model = PolicyValueNet.from_config(model_config).to(device)
    model.load_state_dict(data["model"])
    optimizer = make_optimizer(model, config)
    optimizer.load_state_dict(data["optimizer"])
    return TrainState(
        model=model,
        optimizer=optimizer,
        iteration=data["iteration"],
        step=data.get("step", 0),
        trained_until=data.get("trained_until", 0.0),
    )


def current_iteration(config: Config) -> int:
    data = torch.load(config.paths.checkpoint_dir / LATEST_CHECKPOINT, map_location="cpu", weights_only=True)
    return data["iteration"]


def init(config: Config) -> None:
    model = PolicyValueNet.from_config(config.model)
    save(config, TrainState(model, make_optimizer(model, config), iteration=0, step=0, trained_until=0.0))


def new_positions(selfplay_dir: Path, since: float) -> tuple[int, float]:
    """更新時刻が since より新しい自己対局ファイルの局面数と、その中で最も新しい更新時刻。"""
    count, newest = 0, since
    for path in selfplay_dir.glob("*.npz"):
        mtime = path.stat().st_mtime
        if mtime > since:
            with np.load(path) as data:
                count += len(data["value"])
            newest = max(newest, mtime)
    return count, newest


def train_iteration(config: Config) -> int | None:
    """新しい自己対局データの量に応じて学習し、新しいイテレーション番号を返す。学習しなかったら None。"""
    tc = config.train
    device = torch.device(tc.device)
    state = load(config, device)
    fresh, newest = new_positions(config.paths.selfplay_dir, state.trained_until)
    if fresh == 0:
        print("新しい自己対局データがないので学習しません")
        return None
    buffer = ReplayBuffer.load(config.paths.selfplay_dir, tc.replay_window, device)
    if len(buffer) < tc.min_replay:
        print(f"リプレイバッファが {len(buffer)} / {tc.min_replay} 局面なので、まだ学習しません")
        return None

    steps = math.ceil(fresh * tc.samples_per_position / tc.batch_size)
    iteration = state.iteration + 1
    print(f"iteration {iteration}: 新しい局面 {fresh} / バッファ {len(buffer)} 局面で {steps} ステップ学習します")

    writer = SummaryWriter(config.paths.log_dir)
    use_amp = tc.amp and device.type == "cuda"
    model, optimizer = state.model, state.optimizer
    model.train()
    for i in tqdm(range(steps), desc=f"iter {iteration}", mininterval=5):
        lr = lr_at(tc.lr_schedule, state.step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        states, target_policy, target_value = buffer.sample(tc.batch_size)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            policy_logits, value = model(states)
        loss, policy_loss, value_loss = loss_fn(
            policy_logits, value, target_policy, target_value, tc.value_loss_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        state.step += 1

        if i % 50 == 0 or i == steps - 1:
            writer.add_scalar("loss/total", loss.item(), state.step)
            writer.add_scalar("loss/policy", policy_loss.item(), state.step)
            writer.add_scalar("loss/value", value_loss.item(), state.step)
            writer.add_scalar("train/lr", lr, state.step)
    writer.add_scalar("train/iteration", iteration, state.step)
    writer.add_scalar("train/replay_size", len(buffer), state.step)
    writer.close()
    print(f"最終損失: policy {policy_loss.item():.4f} / value {value_loss.item():.4f} (lr {lr:g})")

    state.iteration = iteration
    state.trained_until = newest
    save(config, state)
    del buffer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return iteration


def prune(config: Config, latest: int) -> None:
    """直近 keep_last 個・keep_every ごと・強さの評価に使う基準モデル以外の古いファイルを消す。"""
    lc = config.loop
    interval = config.evaluate.interval
    last_eval = latest - latest % interval
    protected = {last_eval, last_eval - interval}

    def keep(i: int) -> bool:
        return i > latest - lc.keep_last or i % lc.keep_every == 0 or i in protected

    for directory, suffix in ((config.paths.checkpoint_dir, ".pt"), (config.paths.model_dir, ".onnx")):
        for path in directory.glob(f"iter_*{suffix}"):
            match = ITER_PATTERN.fullmatch(path.name.removesuffix(suffix))
            if match and not keep(int(match.group(1))):
                path.unlink()
