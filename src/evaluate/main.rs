//! 2 つのモデルを対戦させて強さを比べる。先後は 1 局ごとに入れ替える。
//!
//!     cargo run --release --bin evaluate -- --model new.onnx --opponent old.onnx [--games N] [--summary out.json]

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

use anyhow::bail;
use clap::Parser;
use gomoku_engine::config::Config;
use gomoku_engine::inference::InferenceServer;
use gomoku_engine::selfplay::play_match;
use gomoku_engine::{GameResult, Mcts, Stone};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "config/config.toml")]
    config: PathBuf,
    /// 評価するモデル
    #[arg(long)]
    model: PathBuf,
    /// 対戦相手のモデル
    #[arg(long)]
    opponent: PathBuf,
    /// 対局数 (省略時は設定ファイルの evaluate.games)
    #[arg(long)]
    games: Option<usize>,
    /// 集計結果を JSON で書き出す先
    #[arg(long)]
    summary: Option<PathBuf>,
}

/// model から見た成績
#[derive(Default, Serialize)]
struct Summary {
    games: usize,
    wins: usize,
    losses: usize,
    draws: usize,
    black_games: usize,
    black_wins: usize,
    white_games: usize,
    white_wins: usize,
    /// (勝ち + 引き分け / 2) / 対局数
    score: f64,
    /// 相手に対する Elo レーティング差の推定値
    elo_diff: f64,
}

impl Summary {
    fn add(&mut self, model_color: Stone, result: GameResult) {
        self.games += 1;
        let won = result == GameResult::Win(model_color);
        match result {
            GameResult::Win(_) if won => self.wins += 1,
            GameResult::Win(_) => self.losses += 1,
            _ => self.draws += 1,
        }
        match model_color {
            Stone::Black => {
                self.black_games += 1;
                self.black_wins += won as usize;
            }
            Stone::White => {
                self.white_games += 1;
                self.white_wins += won as usize;
            }
        }
        self.score = (self.wins as f64 + 0.5 * self.draws as f64) / self.games as f64;
        // 全勝・全敗で無限大にならないよう 0.5 局分だけ内側に寄せる
        let margin = 0.5 / self.games as f64;
        let s = self.score.clamp(margin, 1.0 - margin);
        self.elo_diff = -400.0 * (1.0 / s - 1.0).log10();
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = Config::load(&args.config)?;
    let games = args.games.unwrap_or(config.evaluate.games);
    let mcts_config = config.evaluate.mcts(&config.selfplay);
    let temperature_moves = config.evaluate.temperature_moves;

    let model = InferenceServer::load(&args.model, &config.inference)?;
    let opponent = InferenceServer::load(&args.opponent, &config.inference)?;
    println!("{} vs {} ({games} 局)", args.model.display(), args.opponent.display());

    let next_game = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel::<(Stone, GameResult)>();
    let workers: Vec<_> = (0..config.selfplay.workers.min(games))
        .map(|_| {
            let (mut model, mut opponent) = (model.evaluator(), opponent.evaluator());
            let tx = tx.clone();
            let next_game = Arc::clone(&next_game);
            let mut mcts = Mcts::new(mcts_config.clone());
            thread::spawn(move || {
                let mut rng = StdRng::from_os_rng();
                loop {
                    let game = next_game.fetch_add(1, Ordering::Relaxed);
                    if game >= games {
                        break;
                    }
                    let (color, result) = if game % 2 == 0 {
                        (Stone::Black, play_match(&mut mcts, &mut model, &mut opponent, temperature_moves, &mut rng))
                    } else {
                        (Stone::White, play_match(&mut mcts, &mut opponent, &mut model, temperature_moves, &mut rng))
                    };
                    if tx.send((color, result)).is_err() {
                        break;
                    }
                }
            })
        })
        .collect();
    drop(tx);

    let mut summary = Summary::default();
    for (color, result) in rx {
        summary.add(color, result);
    }
    let panicked = workers.into_iter().map(|w| w.join()).filter(Result::is_err).count();
    model.shutdown()?;
    opponent.shutdown()?;
    if panicked > 0 {
        bail!("{panicked} 個の対局スレッドが異常終了しました");
    }

    println!(
        "{} 勝 {} 敗 {} 分 (黒番 {}/{} 勝, 白番 {}/{} 勝) | 勝率 {:.1}% | Elo 差 {:+.0}",
        summary.wins,
        summary.losses,
        summary.draws,
        summary.black_wins,
        summary.black_games,
        summary.white_wins,
        summary.white_games,
        summary.score * 100.0,
        summary.elo_diff
    );
    if let Some(path) = args.summary {
        std::fs::write(path, serde_json::to_string_pretty(&summary)?)?;
    }
    Ok(())
}
