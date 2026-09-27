//! config/config.toml の読み込み (Rust 側で使う項目のみ)。

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;

use crate::mcts::MctsConfig;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub paths: Paths,
    pub selfplay: SelfPlayConfig,
    pub evaluate: EvaluateConfig,
    pub inference: InferenceConfig,
}

#[derive(Debug, Deserialize)]
pub struct Paths {
    pub selfplay_dir: PathBuf,
    pub model_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
pub struct SelfPlayConfig {
    pub games: usize,
    pub workers: usize,
    pub simulations: u32,
    pub mcts_batch_size: usize,
    pub c_puct: f32,
    pub fpu_reduction: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
    pub temperature_moves: usize,
    pub games_per_file: usize,
}

impl SelfPlayConfig {
    pub fn mcts(&self) -> MctsConfig {
        MctsConfig {
            simulations: self.simulations,
            batch_size: self.mcts_batch_size,
            c_puct: self.c_puct,
            fpu_reduction: self.fpu_reduction,
            dirichlet_alpha: self.dirichlet_alpha,
            dirichlet_epsilon: self.dirichlet_epsilon,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EvaluateConfig {
    pub games: usize,
    pub simulations: u32,
    pub temperature_moves: usize,
}

impl EvaluateConfig {
    /// 自己対局の探索設定をもとに、探索回数を変えてノイズを切ったもの。
    pub fn mcts(&self, selfplay: &SelfPlayConfig) -> MctsConfig {
        MctsConfig { simulations: self.simulations, dirichlet_epsilon: 0.0, ..selfplay.mcts() }
    }
}

#[derive(Debug, Deserialize)]
pub struct InferenceConfig {
    pub device: Device,
    pub max_batch: usize,
    /// 推論スレッド数 (それぞれがモデルを読み込む)
    pub threads: usize,
    /// libonnxruntime.so のあるディレクトリ
    pub onnxruntime_dir: PathBuf,
    #[serde(default)]
    pub cuda_lib_dirs: Vec<PathBuf>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Device {
    Cpu,
    Cuda,
}

impl Config {
    /// 相対パスはプロジェクトルート (設定ファイルから上にたどって最初に pyproject.toml があるディレクトリ。
    /// 見つからなければカレントディレクトリ) からのパスとして解決する。src/train/config.py と同じ規則。
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("{} を読めません", path.display()))?;
        let mut config: Config = toml::from_str(&text).with_context(|| format!("{} の形式が不正です", path.display()))?;
        let path = path.canonicalize()?;
        let root = match path.ancestors().skip(1).find(|dir| dir.join("pyproject.toml").exists()) {
            Some(dir) => dir.to_path_buf(),
            None => std::env::current_dir()?,
        };
        let resolve = |p: &mut PathBuf| *p = root.join(&*p);
        resolve(&mut config.paths.selfplay_dir);
        resolve(&mut config.paths.model_dir);
        resolve(&mut config.inference.onnxruntime_dir);
        config.inference.cuda_lib_dirs.iter_mut().for_each(resolve);
        Ok(config)
    }

    pub fn latest_model(&self) -> PathBuf {
        self.paths.model_dir.join("latest.onnx")
    }
}
