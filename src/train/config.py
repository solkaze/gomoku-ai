"""config/config.toml の読み込み。"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.toml"


@dataclass(frozen=True)
class Paths:
    selfplay_dir: Path
    checkpoint_dir: Path
    model_dir: Path
    log_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    channels: int
    blocks: int


@dataclass(frozen=True)
class TrainConfig:
    device: str
    batch_size: int
    samples_per_position: float
    min_replay: int
    lr_schedule: list[tuple[int, float]]
    weight_decay: float
    replay_window: int
    value_loss_weight: float
    amp: bool
    export_fp16: bool


@dataclass(frozen=True)
class LoopConfig:
    selfplay_bin: Path
    evaluate_bin: Path
    stop_file: Path
    keep_last: int
    keep_every: int


@dataclass(frozen=True)
class EvaluateConfig:
    interval: int
    games: int
    simulations: int
    temperature_moves: int


@dataclass(frozen=True)
class Config:
    path: Path
    paths: Paths
    model: ModelConfig
    train: TrainConfig
    loop: LoopConfig
    evaluate: EvaluateConfig


def find_root(config_path: Path) -> Path:
    """設定ファイル中の相対パスの基準。設定ファイルから上にたどって最初に pyproject.toml があるディレクトリ、
    見つからなければカレントディレクトリ。src/engine/config.rs と同じ規則。"""
    for directory in config_path.resolve().parents:
        if (directory / "pyproject.toml").exists():
            return directory
    return Path.cwd()


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    root = find_root(path)
    paths = Paths(**{k: root / v for k, v in raw["paths"].items()})
    train = dict(raw["train"])
    train["lr_schedule"] = sorted((int(step), float(lr)) for step, lr in train["lr_schedule"])
    loop = dict(raw["loop"])
    for key in ("selfplay_bin", "evaluate_bin", "stop_file"):
        loop[key] = root / loop[key]
    return Config(
        path=path.resolve(),
        paths=paths,
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**train),
        loop=LoopConfig(**loop),
        evaluate=EvaluateConfig(**raw["evaluate"]),
    )
