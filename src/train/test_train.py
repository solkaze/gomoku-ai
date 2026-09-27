"""uv run pytest src/train"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from config import DEFAULT_CONFIG, Paths, load_config
from export import export_onnx
from network import BOARD_SIZE, CELLS, IN_CHANNELS, PolicyValueNet, loss_fn
from replay import ReplayBuffer, augment, transform
from trainer import lr_at, new_positions, prune


def fake_data(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = rng.integers(0, 2, (n, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    policy = rng.random((n, CELLS), dtype=np.float32)
    policy /= policy.sum(axis=1, keepdims=True)
    value = rng.choice(np.array([-1.0, 0.0, 1.0], dtype=np.float32), n)
    return states, policy, value


def test_eight_distinct_symmetries() -> None:
    x = torch.zeros(1, 1, BOARD_SIZE, BOARD_SIZE)
    x[0, 0, 1, 3] = 1  # 対称軸上にない点
    images = {tuple(transform(x, s).nonzero()[0].tolist()) for s in range(8)}
    assert len(images) == 8


def test_augment_moves_policy_with_stones() -> None:
    n = 256
    states = torch.zeros(n, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    policy = torch.zeros(n, 1, BOARD_SIZE, BOARD_SIZE)
    states[:, 0, 2, 5] = 1
    policy[:, 0, 2, 5] = 1
    states, policy = augment(states, policy)
    assert torch.equal(states[:, 0], policy[:, 0])
    # 元の入力は変更しない
    assert policy.sum() == n


def test_training_reduces_loss() -> None:
    torch.manual_seed(0)
    model = PolicyValueNet(channels=16, blocks=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    states, policy, value = (torch.from_numpy(a) for a in fake_data(32))
    states = states.float()

    def step() -> float:
        logits, v = model(states)
        loss, _, _ = loss_fn(logits, v, policy, value, 1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    first = step()
    for _ in range(50):
        last = step()
    assert last < first


@pytest.mark.parametrize(("fp16", "atol"), [(False, 1e-4), (True, 2e-2)])
def test_onnx_matches_torch(tmp_path: Path, fp16: bool, atol: float) -> None:
    torch.manual_seed(0)
    model = PolicyValueNet(channels=16, blocks=2).eval()
    path = tmp_path / "model.onnx"
    export_onnx(model, path, fp16=fp16)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    for batch in (1, 5):  # 書き出し時と異なるバッチサイズでも動く
        x = torch.from_numpy(fake_data(batch)[0]).float()
        policy, value = session.run(["policy", "value"], {"input": x.numpy()})
        with torch.no_grad():
            expected_policy, expected_value = model(x)
        assert policy.shape == (batch, CELLS)
        assert value.shape == (batch,)
        assert policy.dtype == value.dtype == np.float32
        np.testing.assert_allclose(policy, expected_policy.numpy(), atol=atol)
        np.testing.assert_allclose(value, expected_value.numpy(), atol=atol)


def test_replay_keeps_newest_window(tmp_path: Path) -> None:
    for i, n in enumerate([30, 30, 30]):
        states, policy, value = fake_data(n, seed=i)
        value[:] = i  # どのファイル由来か分かるように
        path = tmp_path / f"{i}.npz"
        np.savez(path, states=states, policy=policy, value=value)
        os.utime(path, (i, i))

    buffer = ReplayBuffer.load(tmp_path, window=50, device=torch.device("cpu"))
    assert len(buffer) == 50
    assert buffer.value[:20].eq(1).all() and buffer.value[20:].eq(2).all()

    states, policy, value = buffer.sample(8)
    assert states.shape == (8, IN_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert policy.shape == (8, CELLS)
    assert value.shape == (8,)


def test_replay_rejects_wrong_format(tmp_path: Path) -> None:
    states, policy, value = fake_data(4)
    np.savez(tmp_path / "bad.npz", states=states.astype(np.float32), policy=policy, value=value)
    with pytest.raises(ValueError, match="states"):
        ReplayBuffer.load(tmp_path, window=10, device=torch.device("cpu"))


def test_lr_schedule() -> None:
    schedule = [(0, 1e-3), (100, 3e-4), (300, 1e-4)]
    assert lr_at(schedule, 0) == 1e-3
    assert lr_at(schedule, 99) == 1e-3
    assert lr_at(schedule, 100) == 3e-4
    assert lr_at(schedule, 10_000) == 1e-4


def test_new_positions_counts_only_newer_files(tmp_path: Path) -> None:
    for i, n in enumerate([10, 20, 30]):
        path = tmp_path / f"{i}.npz"
        np.savez(path, value=np.zeros(n, dtype=np.float32))
        os.utime(path, (100 + i, 100 + i))
    assert new_positions(tmp_path, since=0) == (60, 102)
    assert new_positions(tmp_path, since=100) == (50, 102)
    assert new_positions(tmp_path, since=102) == (0, 102)


def test_prune_keeps_recent_periodic_and_eval_reference(tmp_path: Path) -> None:
    config = load_config()
    paths = Paths(tmp_path / "sp", tmp_path / "ckpt", tmp_path / "models", tmp_path / "logs")
    config = replace(
        config,
        paths=paths,
        loop=replace(config.loop, keep_last=2, keep_every=10),
        evaluate=replace(config.evaluate, interval=4),
    )
    for d in (paths.checkpoint_dir, paths.model_dir):
        d.mkdir()
        for i in range(24):
            (d / f"iter_{i:04d}{'.pt' if d == paths.checkpoint_dir else '.onnx'}").touch()
        (d / "latest.pt").touch()
    prune(config, latest=23)
    kept = sorted(int(p.stem[5:]) for p in paths.model_dir.glob("iter_*.onnx"))
    # 10 ごと (0, 10, 20) + 直近 2 個 (22, 23) + 評価の基準 (20 と 1 つ前の 16)
    assert kept == [0, 10, 16, 20, 22, 23]
    assert sorted(int(p.stem[5:]) for p in paths.checkpoint_dir.glob("iter_*.pt")) == kept
    assert (paths.checkpoint_dir / "latest.pt").exists()


def test_relative_paths_are_resolved_from_project_root(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").touch()
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "config.toml").write_text(DEFAULT_CONFIG.read_text())
    config = load_config(nested / "config.toml")
    assert config.paths.model_dir == tmp_path.resolve() / "runs" / "models"
    assert config.loop.selfplay_bin == tmp_path.resolve() / "target" / "release" / "selfplay"
