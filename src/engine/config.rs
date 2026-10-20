//! config/config.toml の読み込み (Rust 側で使う項目のみ)。

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;

use crate::mcts::MctsConfig;
use crate::selfplay::{OpeningBalance, PlayoutCap};

#[derive(Debug, Deserialize)]
pub struct Config {
    pub paths: Paths,
    pub selfplay: SelfPlayConfig,
    pub evaluate: EvaluateConfig,
    pub inference: InferenceConfig,
    pub play: PlayConfig,
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
    /// 開局でランダムに打つ手数の上限 (0 で無効)
    pub opening_moves: usize,
    /// 互角の開局: opening_balance_games の割合の局で、開局の局面を opening_balance_simulations 回の探索で評価し、
    /// 手番側から見た価値の絶対値がこれ以下のものだけを使う (selfplay::OpeningBalance)。1.0 で無効
    pub opening_balance: f32,
    pub opening_balance_simulations: u32,
    pub opening_balance_games: f64,
    /// 探索のルートで VCF を読む (mcts::MctsConfig::vcf)。評価 (evaluate) にも同じ値を使う
    pub vcf: bool,
    /// playout cap randomization: 通常の探索 (simulations 回) をする手の割合。1.0 で無効
    pub full_search_prob: f64,
    /// それ以外の手の探索回数 (ノイズなし・学習データにしない)
    pub fast_simulations: u32,
    pub games_per_file: usize,
}

impl SelfPlayConfig {
    pub fn playout_cap(&self) -> PlayoutCap {
        PlayoutCap { full_prob: self.full_search_prob, fast_simulations: self.fast_simulations }
    }

    pub fn opening_balance(&self) -> OpeningBalance {
        OpeningBalance {
            max_value: self.opening_balance,
            simulations: self.opening_balance_simulations,
            games: self.opening_balance_games,
        }
    }

    pub fn mcts(&self) -> MctsConfig {
        MctsConfig {
            simulations: self.simulations,
            batch_size: self.mcts_batch_size,
            c_puct: self.c_puct,
            fpu_reduction: self.fpu_reduction,
            dirichlet_alpha: self.dirichlet_alpha,
            dirichlet_epsilon: self.dirichlet_epsilon,
            vcf: self.vcf,
            moves_left_effect: 0.0,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EvaluateConfig {
    pub games: usize,
    pub simulations: u32,
    pub temperature_moves: usize,
    /// 開局でランダムに打つ手数の上限 (0 で無効)
    pub opening_moves: usize,
}

impl EvaluateConfig {
    /// 自己対局の探索設定をもとに、探索回数を変えてノイズを切ったもの。
    pub fn mcts(&self, selfplay: &SelfPlayConfig) -> MctsConfig {
        MctsConfig { simulations: self.simulations, dirichlet_epsilon: 0.0, ..selfplay.mcts() }
    }
}

#[derive(Debug, Deserialize)]
pub struct PlayConfig {
    /// 人と AI の対戦 (gomoku) の記録を書き出す先
    pub record_dir: PathBuf,
    /// 負けの局面で終局を遠ざけ、勝ちの局面で早く終わらせる強さ (mcts::MctsConfig::moves_left_effect)
    pub moves_left_effect: f32,
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
        resolve(&mut config.play.record_dir);
        resolve(&mut config.inference.onnxruntime_dir);
        config.inference.cuda_lib_dirs.iter_mut().for_each(resolve);
        Ok(config)
    }

    pub fn latest_model(&self) -> PathBuf {
        self.paths.model_dir.join("latest.onnx")
    }

    /// 人との対戦に使うモデル: ゲーティングで選ばれた最強モデル (best.onnx)。まだなければ最新モデル。
    pub fn best_model(&self) -> PathBuf {
        let best = self.paths.model_dir.join("best.onnx");
        if best.exists() { best } else { self.latest_model() }
    }
}
