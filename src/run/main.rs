//! 人と AI の対戦。
//!
//!     cargo run --release --bin gomoku -- [--model path.onnx] [--color white] [--simulations N] [--cpu] [--plain]
//!
//! 端末では盤を色付きで描き、矢印キーかマウスで打つ (tui.rs)。--plain を付けるか、入出力が端末でなければ
//! 座標を「x y」で入力する行単位の表示になる (plain.rs)。
//! 座標は x が左から、y が上から、どちらも 0〜14 で、天元は 7 7。

mod plain;
mod tui;

use std::io::{self, IsTerminal};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use clap::{Parser, ValueEnum};
use gomoku_engine::config::{Config, Device, InferenceConfig};
use gomoku_engine::inference::{LoadStage, OnnxModel, init_runtime_with};
use gomoku_engine::{Board, GameResult, Mcts, MctsConfig, MoveError, SIZE, SearchResult, Stone};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Clone, Copy, ValueEnum)]
enum Color {
    Black,
    White,
}

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "config/config.toml")]
    config: PathBuf,
    /// 使うモデル (省略時は model_dir/latest.onnx)
    #[arg(long)]
    model: Option<PathBuf>,
    /// 人が持つ石 (省略時は起動後に尋ねる)
    #[arg(long, value_enum)]
    color: Option<Color>,
    /// AI の 1 手あたりの探索回数
    #[arg(long, default_value_t = 4000)]
    simulations: u32,
    /// 設定によらず CPU で推論する
    #[arg(long)]
    cpu: bool,
    /// AI の読みの候補手を何手まで表示するか (0 で表示しない)
    #[arg(long, default_value_t = 5)]
    candidates: usize,
    /// 盤を色付きで描かず、座標を入力する行単位の表示にする
    #[arg(long)]
    plain: bool,
}

/// 対局の設定 (引数と設定ファイルから決まるもの)
struct Setup {
    model_path: PathBuf,
    inference: InferenceConfig,
    mcts: MctsConfig,
    /// 人が持つ石 (None なら起動後に尋ねる)
    color: Option<Stone>,
    candidates: usize,
}

impl Setup {
    /// モデルのファイル名と推論の設定 (画面の見出し用)
    fn describe(&self) -> String {
        let name = self.model_path.file_name().unwrap_or(self.model_path.as_os_str()).to_string_lossy();
        format!("{name} ({:?}, 探索 {} 回/手)", self.inference.device, self.mcts.simulations)
    }
}

/// モデルと探索木。AI の手とヒントの両方に使う。
struct Engine {
    model: OnnxModel,
    mcts: Mcts,
    rng: StdRng,
}

impl Engine {
    /// 読み込みの各段階に入るたびに on_stage を呼ぶ。
    fn load(setup: &Setup, on_stage: &mut dyn FnMut(LoadStage)) -> anyhow::Result<Self> {
        init_runtime_with(&setup.inference, on_stage)?;
        let model = OnnxModel::load_with(&setup.model_path, setup.inference.device, on_stage)?;
        Ok(Self { model, mcts: Mcts::new(setup.mcts.clone()), rng: StdRng::from_os_rng() })
    }

    fn search(&mut self, board: &Board) -> SearchResult {
        self.search_until(board, &AtomicBool::new(false))
    }

    /// stop が立ったら探索を打ち切る。
    fn search_until(&mut self, board: &Board, stop: &AtomicBool) -> SearchResult {
        self.mcts.search_until(board, &mut self.model, &mut self.rng, stop)
    }
}

fn coord(p: usize) -> String {
    format!("{} {}", p % SIZE, p / SIZE)
}

/// 訪問回数の多い順に n 手の候補と、その訪問回数の割合 (%)。
fn top_candidates(result: &SearchResult, n: usize) -> Vec<(usize, f64)> {
    let total: u32 = result.visits.iter().map(|&(_, n)| n).sum();
    let mut visits = result.visits.clone();
    visits.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    visits.into_iter().take(n).map(|(mv, n)| (mv, n as f64 * 100.0 / total.max(1) as f64)).collect()
}

/// 棋譜を残しながら進める対局。
struct Game {
    board: Board,
    moves: Vec<usize>,
}

impl Game {
    fn new() -> Self {
        Self { board: Board::new(), moves: Vec::new() }
    }

    fn record(&self) -> String {
        let record: Vec<String> = self.moves.iter().enumerate().map(|(i, &p)| format!("{}.({})", i + 1, coord(p))).collect();
        record.join(" ")
    }

    fn play(&mut self, p: usize) -> Result<GameResult, MoveError> {
        let result = self.board.play(p)?;
        self.moves.push(p);
        Ok(result)
    }

    /// human の直前の手を打つ前まで戻す。戻せる手がなければ false。
    fn undo(&mut self, human: Stone) -> bool {
        // パスはないので、黒の手は偶数番目 (0 始まり)、白の手は奇数番目
        let parity = if human == Stone::Black { 0 } else { 1 };
        let Some(last) = (0..self.moves.len()).rev().find(|i| i % 2 == parity) else {
            return false;
        };
        self.moves.truncate(last);
        self.board = Board::new();
        for &p in &self.moves {
            self.board.play(p).expect("棋譜の手は合法手");
        }
        true
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = Config::load(&args.config)?;
    let model_path = args.model.unwrap_or_else(|| config.latest_model());
    let mut inference = config.inference;
    if args.cpu {
        inference.device = Device::Cpu;
    }
    let setup = Setup {
        model_path,
        inference,
        mcts: MctsConfig { simulations: args.simulations, dirichlet_epsilon: 0.0, ..config.selfplay.mcts() },
        color: args.color.map(|c| match c {
            Color::Black => Stone::Black,
            Color::White => Stone::White,
        }),
        candidates: args.candidates,
    };
    if args.plain || !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        plain::run(setup)
    } else {
        tui::run(setup)
    }
}
