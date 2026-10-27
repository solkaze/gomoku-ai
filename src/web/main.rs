//! ブラウザで AI と対戦する Web サーバー。
//!
//!     cargo run --release --bin web -- [--addr 127.0.0.1:39115] [--model path.onnx] [--simulations N] [--cpu]
//!
//! 対局の状態はブラウザが持ち、手を打つたびに棋譜を送る。サーバーは棋譜から局面を作り直し、AI の手番なら探索して
//! 1 手打ってから、新しい局面 (棋譜・勝敗・黒の禁点) を返す。黒の初手は天元に固定。
//! 探索は 1 局面ずつ順に行う (GPU を学習と分け合うため)。モデルのファイルが置き換わったら (学習のゲーティングで
//! best.onnx が更新されたら)、次の探索の前に読み直す。
//! 待ち受けるアドレスの既定は 127.0.0.1 (このマシンからだけ開ける)。誰でも開ける場所に出すことがあるので、
//! 探索の順番待ちが MAX_WAITING を超えたら新しい要求は断る (GPU を使い切られないように)。

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use anyhow::Context;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use gomoku_engine::config::{Config, Device, InferenceConfig};
use gomoku_engine::inference::{self, OnnxModel, init_runtime};
use gomoku_engine::{Board, CELLS, CENTER, GameResult, Mcts, MctsConfig, Stone};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

const PAGE: &str = include_str!("index.html");

/// 同時に受け付ける要求の数 (探索中のものを含む)。これを超えたら「混み合っています」と断る
const MAX_WAITING: usize = 4;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "config/config.toml")]
    config: PathBuf,
    /// 待ち受けるアドレス (Docker のコンテナ内では 0.0.0.0 にし、公開範囲は compose の ports で絞る)
    #[arg(long, default_value = "127.0.0.1:39115")]
    addr: SocketAddr,
    /// 使うモデル (省略時は model_dir/best.onnx。まだなければ latest.onnx)
    #[arg(long)]
    model: Option<PathBuf>,
    /// AI の 1 手あたりの探索回数
    #[arg(long, default_value_t = 16000)]
    simulations: u32,
    /// 設定によらず CPU で推論する
    #[arg(long)]
    cpu: bool,
}

/// モデルと探索木。モデルのファイルが置き換わったら読み直す。
struct Ai {
    path: PathBuf,
    device: Device,
    /// 読み込んだときのファイルの更新時刻
    modified: Option<SystemTime>,
    model: OnnxModel,
    mcts: Mcts,
    rng: StdRng,
}

impl Ai {
    fn load(path: PathBuf, inference: &InferenceConfig, mcts: MctsConfig) -> anyhow::Result<Self> {
        init_runtime(inference)?;
        let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        let model = OnnxModel::load(&path, inference.device)?;
        Ok(Self { path, device: inference.device, modified, model, mcts: Mcts::new(mcts), rng: StdRng::from_os_rng() })
    }

    /// ファイルが置き換わっていたら読み直す。読めなければ今のモデルを使い続ける。
    fn reload_if_changed(&mut self) {
        let modified = std::fs::metadata(&self.path).and_then(|m| m.modified()).ok();
        if modified.is_none() || modified == self.modified {
            return;
        }
        match OnnxModel::load(&self.path, self.device) {
            Ok(model) => {
                self.model = model;
                self.modified = modified;
                println!("モデルを読み直しました: {}", self.path.display());
            }
            Err(e) => eprintln!("モデルを読み直せませんでした (今のモデルを使い続けます): {e:?}"),
        }
    }

    /// 手番側の手と、手番側から見た評価値 [-1, 1]。
    fn think(&mut self, board: &Board) -> (usize, f32) {
        self.reload_if_changed();
        let result = self.mcts.search(board, &mut self.model, &mut self.rng);
        (result.best_move(), result.value)
    }
}

struct App {
    ai: Mutex<Ai>,
    info: Info,
    /// 受け付けて、まだ応答していない要求の数
    waiting: AtomicUsize,
}

/// 受け付けた要求を数え、応答したら (落とされても) 数から外す。
struct Waiting<'a>(&'a AtomicUsize);

impl<'a> Waiting<'a> {
    fn enter(count: &'a AtomicUsize) -> Option<Self> {
        let guard = Self(count);
        (count.fetch_add(1, Ordering::SeqCst) < MAX_WAITING).then_some(guard)
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Clone, Serialize)]
struct Info {
    model: String,
    simulations: u32,
    device: String,
}

#[derive(Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Color {
    Black,
    White,
}

impl From<Stone> for Color {
    fn from(stone: Stone) -> Self {
        match stone {
            Stone::Black => Color::Black,
            Stone::White => Color::White,
        }
    }
}

#[derive(Deserialize)]
struct MoveRequest {
    /// ここまでの棋譜 (0 始まりの点の番号 y * 15 + x)。空なら黒の初手 (天元) を打つ
    moves: Vec<usize>,
    /// AI が持つ石
    ai: Color,
}

#[derive(Serialize)]
struct MoveResponse {
    moves: Vec<usize>,
    /// "ongoing" / "black" / "white" (勝った側) / "draw"
    result: &'static str,
    /// 黒の禁じ手で決着したか
    lost_by_forbidden: bool,
    to_move: Color,
    /// 黒番のときの黒の禁点
    forbidden: Vec<usize>,
    /// AI がこの応答で打った手と、AI から見た勝率 (%)
    ai_move: Option<usize>,
    ai_win_rate: Option<f64>,
}

struct ApiError(StatusCode, String);

impl ApiError {
    fn bad_request(message: String) -> Self {
        Self(StatusCode::BAD_REQUEST, message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(serde_json::json!({ "error": self.1 }))).into_response()
    }
}

/// 棋譜を並べ直し、AI の手番なら 1 手打って、新しい局面を返す。
fn respond(ai: &Mutex<Ai>, request: MoveRequest) -> Result<MoveResponse, ApiError> {
    let mut board = Board::new();
    let mut moves = request.moves;
    if moves.is_empty() {
        moves.push(CENTER);
    }
    for (i, &p) in moves.iter().enumerate() {
        if p >= CELLS {
            return Err(ApiError::bad_request(format!("{} 手目が盤外です", i + 1)));
        }
        board
            .play(p)
            .map_err(|e| ApiError::bad_request(format!("{} 手目 ({} {}) には打てません: {e}", i + 1, p % 15, p / 15)))?;
    }
    let (mut ai_move, mut ai_win_rate) = (None, None);
    if board.result() == GameResult::Ongoing && Color::from(board.side_to_move()) == request.ai {
        let (p, value) = ai.lock().unwrap_or_else(|e| e.into_inner()).think(&board);
        board.play(p).expect("探索結果は合法手");
        moves.push(p);
        ai_move = Some(p);
        ai_win_rate = Some(((value as f64 + 1.0) * 50.0 * 10.0).round() / 10.0);
    }
    let result = match board.result() {
        GameResult::Ongoing => "ongoing",
        GameResult::Win(Stone::Black) => "black",
        GameResult::Win(Stone::White) => "white",
        GameResult::Draw => "draw",
    };
    let forbidden = if board.result() == GameResult::Ongoing && board.side_to_move() == Stone::Black {
        board.forbidden_mask().iter().enumerate().filter(|&(_, &f)| f).map(|(p, _)| p).collect()
    } else {
        Vec::new()
    };
    Ok(MoveResponse {
        moves,
        result,
        lost_by_forbidden: board.lost_by_forbidden(),
        to_move: board.side_to_move().into(),
        forbidden,
        ai_move,
        ai_win_rate,
    })
}

async fn post_move(State(app): State<Arc<App>>, Json(request): Json<MoveRequest>) -> Result<Json<MoveResponse>, ApiError> {
    // 探索は時間がかかるので、非同期の処理を止めないよう別のスレッドで行う
    tokio::task::spawn_blocking(move || {
        let Some(_waiting) = Waiting::enter(&app.waiting) else {
            return Err(ApiError(StatusCode::SERVICE_UNAVAILABLE, "混み合っています。少し待ってからもう一度打ってください".into()));
        };
        respond(&app.ai, request)
    })
    .await
    .map_err(|e| ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("探索に失敗しました: {e}")))?
    .map(Json)
}

async fn get_info(State(app): State<Arc<App>>) -> Json<Info> {
    Json(app.info.clone())
}

fn main() {
    inference::exit(run())
}

fn run() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = Config::load(&args.config)?;
    let model_path = args.model.unwrap_or_else(|| config.best_model());
    let mut inference = config.inference;
    if args.cpu {
        inference.device = Device::Cpu;
    }
    let mcts = MctsConfig {
        simulations: args.simulations,
        dirichlet_epsilon: 0.0,
        vcf: true,
        moves_left_effect: config.play.moves_left_effect,
        ..config.selfplay.mcts()
    };
    let info = Info {
        model: model_path.file_name().unwrap_or(model_path.as_os_str()).to_string_lossy().into_owned(),
        simulations: args.simulations,
        device: format!("{:?}", inference.device),
    };
    println!("モデル: {} ({}, 探索 {} 回/手)", model_path.display(), info.device, info.simulations);
    let ai = Ai::load(model_path, &inference, mcts)?;
    let app = Arc::new(App { ai: Mutex::new(ai), info, waiting: AtomicUsize::new(0) });

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let router = Router::new()
            .route("/", get(|| async { Html(PAGE) }))
            .route("/api/info", get(get_info))
            .route("/api/move", post(post_move))
            .with_state(Arc::clone(&app));
        let listener = tokio::net::TcpListener::bind(args.addr).await.with_context(|| format!("{} で待ち受けられません", args.addr))?;
        println!("http://{} で待ち受けています", args.addr);
        axum::serve(listener, router).with_graceful_shutdown(shutdown_signal()).await?;
        anyhow::Ok(())
    })?;
    // 推論のセッションを破棄してから終了する (inference::exit の前提)
    drop(runtime);
    drop(Arc::try_unwrap(app).ok());
    println!("停止しました");
    Ok(())
}

/// Ctrl-C か SIGTERM (docker compose stop) を待つ。
async fn shutdown_signal() {
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(_) => std::future::pending().await,
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = terminate => {}
    }
}
