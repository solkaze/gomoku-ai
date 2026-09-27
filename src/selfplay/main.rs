//! 自己対局で学習データ (npz) を生成する。
//!
//!     cargo run --release --bin selfplay -- [--games N] [--model path.onnx] [--summary out.json]

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::bail;
use clap::Parser;
use gomoku_engine::config::Config;
use gomoku_engine::features::{FEATURE_LEN, PLANES};
use gomoku_engine::inference::InferenceServer;
use gomoku_engine::npz::{NpyArray, NpyData, write_npz};
use gomoku_engine::selfplay::{Game, play_game};
use gomoku_engine::{CELLS, GameResult, Mcts, SIZE, Stone};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "config/config.toml")]
    config: PathBuf,
    /// 打つ局数 (省略時は設定ファイルの selfplay.games)
    #[arg(long)]
    games: Option<usize>,
    /// 使うモデル (省略時は model_dir/latest.onnx)
    #[arg(long)]
    model: Option<PathBuf>,
    /// 集計結果を JSON で書き出す先 (学習ループが読む)
    #[arg(long)]
    summary: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = Config::load(&args.config)?;
    let sp = &config.selfplay;
    let games = args.games.unwrap_or(sp.games);

    let model_path = args.model.unwrap_or_else(|| config.latest_model());
    let server = InferenceServer::load(&model_path, &config.inference)?;
    println!("モデル: {} ({:?} x {})", model_path.display(), config.inference.device, config.inference.threads);

    let next_game = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel::<Game>();
    let workers: Vec<_> = (0..sp.workers.min(games))
        .map(|_| {
            let mut evaluator = server.evaluator();
            let tx = tx.clone();
            let next_game = Arc::clone(&next_game);
            let mut mcts = Mcts::new(sp.mcts());
            let temperature_moves = sp.temperature_moves;
            thread::spawn(move || {
                let mut rng = StdRng::from_os_rng();
                while next_game.fetch_add(1, Ordering::Relaxed) < games {
                    let game = play_game(&mut mcts, &mut evaluator, temperature_moves, &mut rng);
                    if tx.send(game).is_err() {
                        break;
                    }
                }
            })
        })
        .collect();
    drop(tx);

    std::fs::create_dir_all(&config.paths.selfplay_dir)?;
    let mut writer = Writer::new(&config.paths.selfplay_dir);
    let mut stats = Stats::default();
    let start = Instant::now();
    let tty = std::io::stdout().is_terminal();
    for game in rx {
        stats.add(&game);
        writer.push(game);
        let flush = writer.games >= sp.games_per_file;
        if flush {
            writer.flush()?;
        }
        // 端末では 1 行を上書きし、ログファイルにはファイルを書き出すたびに 1 行出す
        let line = stats.line(games, server.evaluated(), start.elapsed().as_secs_f64());
        if tty {
            print!("\r{line}");
            std::io::stdout().flush()?;
        } else if flush {
            println!("{line}");
        }
    }
    writer.flush()?;
    let secs = start.elapsed().as_secs_f64();
    println!("{}{}", if tty { "\r" } else { "" }, stats.line(games, server.evaluated(), secs));

    let panicked = workers.into_iter().map(|w| w.join()).filter(Result::is_err).count();
    let server_evaluated = server.evaluated();
    server.shutdown()?;
    if panicked > 0 {
        bail!("{panicked} 個の対局スレッドが異常終了しました");
    }
    println!("{} 局 / {} 局面を書き出しました: {}", stats.games, stats.positions, config.paths.selfplay_dir.display());
    if let Some(path) = args.summary {
        let summary = Summary {
            games: stats.games,
            positions: stats.positions,
            black_wins: stats.black,
            white_wins: stats.white,
            draws: stats.draw,
            avg_moves: stats.moves as f64 / stats.games.max(1) as f64,
            games_per_sec: stats.games as f64 / secs,
            evals_per_sec: server_evaluated as f64 / secs,
        };
        std::fs::write(path, serde_json::to_string_pretty(&summary)?)?;
    }
    Ok(())
}

/// games_per_file 局ごとに npz にまとめて書き出す。
struct Writer {
    dir: PathBuf,
    prefix: u64,
    files: usize,
    games: usize,
    states: Vec<u8>,
    policy: Vec<f32>,
    value: Vec<f32>,
}

impl Writer {
    fn new(dir: &Path) -> Self {
        let prefix = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
        Self { dir: dir.to_path_buf(), prefix, files: 0, games: 0, states: vec![], policy: vec![], value: vec![] }
    }

    fn push(&mut self, game: Game) {
        for sample in game.samples {
            self.states.extend(sample.features);
            self.policy.extend(sample.policy);
            self.value.push(sample.value);
        }
        self.games += 1;
    }

    fn flush(&mut self) -> anyhow::Result<()> {
        let n = self.value.len();
        if n == 0 {
            return Ok(());
        }
        debug_assert_eq!(self.states.len(), n * FEATURE_LEN);
        let path = self.dir.join(format!("{}_{:03}.npz", self.prefix, self.files));
        write_npz(
            &path,
            &[
                NpyArray { name: "states", shape: vec![n, PLANES, SIZE, SIZE], data: NpyData::U8(&self.states) },
                NpyArray { name: "policy", shape: vec![n, CELLS], data: NpyData::F32(&self.policy) },
                NpyArray { name: "value", shape: vec![n], data: NpyData::F32(&self.value) },
            ],
        )?;
        self.files += 1;
        self.games = 0;
        self.states.clear();
        self.policy.clear();
        self.value.clear();
        Ok(())
    }
}

#[derive(Serialize)]
struct Summary {
    games: usize,
    positions: usize,
    black_wins: usize,
    white_wins: usize,
    draws: usize,
    avg_moves: f64,
    games_per_sec: f64,
    evals_per_sec: f64,
}

#[derive(Default)]
struct Stats {
    games: usize,
    positions: usize,
    moves: usize,
    black: usize,
    white: usize,
    draw: usize,
}

impl Stats {
    fn add(&mut self, game: &Game) {
        self.games += 1;
        self.positions += game.samples.len();
        self.moves += game.moves;
        match game.result {
            GameResult::Win(Stone::Black) => self.black += 1,
            GameResult::Win(Stone::White) => self.white += 1,
            _ => self.draw += 1,
        }
    }

    fn line(&self, total: usize, evaluated: u64, secs: f64) -> String {
        format!(
            "[{}/{total}] 黒勝 {} / 白勝 {} / 分 {} | 平均 {:.1} 手 | {:.2} 局/s | 評価 {:.0}/s",
            self.games,
            self.black,
            self.white,
            self.draw,
            self.moves as f64 / self.games as f64,
            self.games as f64 / secs,
            evaluated as f64 / secs
        )
    }
}
