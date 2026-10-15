//! 人と AI の対戦と、AI を使わない自由対局。
//!
//!     cargo run --release --bin gomoku -- [--model path.onnx] [--color white] [--simulations N] [--cpu] [--plain]
//!     cargo run --release --bin gomoku -- --free
//!     cargo run --release --bin gomoku -- --allow overline,double-three   (黒の禁じ手の一部を無効にする)
//!
//! 端末では盤を色付きで描き、矢印キーかマウスで打つ (tui.rs)。--plain を付けるか、入出力が端末でなければ
//! 座標を「x y」で入力する行単位の表示になる (plain.rs)。
//! 座標は x が左から、y が上から、どちらも 0〜14 で、天元は 7 7。

mod plain;
mod tui;

use std::io::{self, IsTerminal};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use gomoku_engine::config::{Config, Device, InferenceConfig};
use gomoku_engine::inference::{self, LoadStage, OnnxModel, init_runtime_with};
use gomoku_engine::{Board, GameResult, Mcts, MctsConfig, MoveError, Rules, SIZE, SearchResult, Stone};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Clone, Copy, ValueEnum)]
enum Color {
    Black,
    White,
}

/// 黒の禁じ手の種類
#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Forbidden {
    DoubleThree,
    DoubleFour,
    Overline,
}

impl Forbidden {
    const ALL: [Forbidden; 3] = [Forbidden::DoubleThree, Forbidden::DoubleFour, Forbidden::Overline];

    fn name(self) -> &'static str {
        match self {
            Forbidden::DoubleThree => "三三",
            Forbidden::DoubleFour => "四四",
            Forbidden::Overline => "長連",
        }
    }

    fn enabled(self, rules: Rules) -> bool {
        match self {
            Forbidden::DoubleThree => rules.double_three,
            Forbidden::DoubleFour => rules.double_four,
            Forbidden::Overline => rules.overline,
        }
    }

    fn set(self, rules: &mut Rules, on: bool) {
        match self {
            Forbidden::DoubleThree => rules.double_three = on,
            Forbidden::DoubleFour => rules.double_four = on,
            Forbidden::Overline => rules.overline = on,
        }
    }
}

/// 有効な禁じ手の名前を「・」でつないだもの。1 つもなければ None。
fn forbidden_names(rules: Rules) -> Option<String> {
    let names: Vec<&str> = Forbidden::ALL.into_iter().filter(|f| f.enabled(rules)).map(Forbidden::name).collect();
    (!names.is_empty()).then(|| names.join("・"))
}

/// 黒の禁じ手の説明 (対局前の案内用)
fn forbidden_note(rules: Rules) -> String {
    match forbidden_names(rules) {
        Some(names) => format!("{names}は禁じ手 (打つと負け)"),
        None => "禁じ手なし".to_string(),
    }
}

/// AI は全ての禁じ手を有効にして学習しているので、一部を無効にした対局では注意を出す。
const RULES_WARNING: &str = "AI は全ての禁じ手ありで学習しているため、無効にした禁じ手に関わる形は読みが甘くなります";

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
    /// AI を使わず、黒と白の両方を自分で打つ (モデルは読み込まない)
    #[arg(long, conflicts_with = "color")]
    free: bool,
    /// 黒に許す (禁じ手から外す) 手。カンマ区切りで複数指定できる。長連を外すと黒も六以上の連で勝ちになる
    #[arg(long, value_enum, value_delimiter = ',')]
    allow: Vec<Forbidden>,
    /// AI の 1 手あたりの探索回数
    #[arg(long, default_value_t = 16000)]
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

/// 対局の形式
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// 人 (持つ石を示す) と AI の対戦
    VsAi(Stone),
    /// AI を使わず、黒と白の両方を人が打つ
    Free,
}

impl Mode {
    /// stone を人が打つか
    fn is_human(self, stone: Stone) -> bool {
        match self {
            Mode::VsAi(human) => stone == human,
            Mode::Free => true,
        }
    }
}

/// 対局の設定 (引数と設定ファイルから決まるもの)
struct Setup {
    model_path: PathBuf,
    inference: InferenceConfig,
    mcts: MctsConfig,
    /// 対局の形式 (None なら起動後に尋ねる)
    mode: Option<Mode>,
    /// 黒に適用する禁じ手 (色付きの画面では対局前に切り替えられる)
    rules: Rules,
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

/// 時間を「分:秒」で表す。
fn clock(d: Duration) -> String {
    format!("{}:{:02}", d.as_secs() / 60, d.as_secs() % 60)
}

/// 訪問回数の多い順に n 手の候補と、その訪問回数の割合 (%)。
fn top_candidates(result: &SearchResult, n: usize) -> Vec<(usize, f64)> {
    let total: u32 = result.visits.iter().map(|&(_, n)| n).sum();
    let mut visits = result.visits.clone();
    visits.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    visits.into_iter().take(n).map(|(mv, n)| (mv, n as f64 * 100.0 / total.max(1) as f64)).collect()
}

/// 棋譜と時間を残しながら進める対局。
struct Game {
    board: Board,
    moves: Vec<usize>,
    /// 各手にかかった時間 (moves と同じ並び)
    times: Vec<Duration>,
    started: Instant,
    /// 今の手番になった時刻
    turn_started: Instant,
    /// 終局した時刻
    finished: Option<Instant>,
}

impl Game {
    fn new(rules: Rules) -> Self {
        let now = Instant::now();
        Self { board: Board::with_rules(rules), moves: Vec::new(), times: Vec::new(), started: now, turn_started: now, finished: None }
    }

    /// 棋譜と、対局時間・各手にかかった秒数。
    fn record(&self) -> String {
        let record: Vec<String> = self.moves.iter().enumerate().map(|(i, &p)| format!("{}.({})", i + 1, coord(p))).collect();
        let secs: Vec<String> = self.times.iter().map(|t| format!("{:.1}", t.as_secs_f64())).collect();
        format!(
            "棋譜: {}\n時間: 対局 {} (黒 {} / 白 {})  各手の秒数: {}",
            record.join(" "),
            clock(self.elapsed()),
            clock(self.total(Stone::Black)),
            clock(self.total(Stone::White)),
            secs.join(" ")
        )
    }

    /// 対局開始からの時間 (終局後は進まない)
    fn elapsed(&self) -> Duration {
        self.finished.unwrap_or_else(Instant::now) - self.started
    }

    /// 今の手番が考えている時間
    fn thinking(&self) -> Duration {
        if self.finished.is_some() { Duration::ZERO } else { self.turn_started.elapsed() }
    }

    /// stone がここまでに使った時間 (考えている最中の分も含む)
    fn total(&self, stone: Stone) -> Duration {
        // パスはないので、黒の手は偶数番目 (0 始まり)、白の手は奇数番目
        let parity = if stone == Stone::Black { 0 } else { 1 };
        let played: Duration = self.times.iter().skip(parity).step_by(2).sum();
        if self.board.side_to_move() == stone { played + self.thinking() } else { played }
    }

    /// stone の直前の手 (何手目か (0 始まり)、着手、かかった時間)
    fn last_move_by(&self, stone: Stone) -> Option<(usize, usize, Duration)> {
        let parity = if stone == Stone::Black { 0 } else { 1 };
        let i = (0..self.moves.len()).rev().find(|i| i % 2 == parity)?;
        Some((i, self.moves[i], self.times[i]))
    }

    fn play(&mut self, p: usize) -> Result<GameResult, MoveError> {
        let result = self.board.play(p)?;
        let now = Instant::now();
        self.moves.push(p);
        self.times.push(now - self.turn_started);
        self.turn_started = now;
        if result != GameResult::Ongoing {
            self.finished = Some(now);
        }
        Ok(result)
    }

    /// AI との対戦では人の直前の手を打つ前まで、自由対局では 1 手だけ戻す。戻せる手がなければ false。
    fn undo(&mut self, mode: Mode) -> bool {
        let last = match mode {
            Mode::VsAi(human) => self.last_move_by(human).map(|(i, ..)| i),
            // 初手は天元に固定なので戻さない
            Mode::Free => self.moves.len().checked_sub(1).filter(|&i| i > 0),
        };
        let Some(last) = last else {
            return false;
        };
        self.moves.truncate(last);
        self.times.truncate(last);
        self.board = Board::with_rules(self.board.rules());
        for &p in &self.moves {
            self.board.play(p).expect("棋譜の手は合法手");
        }
        let now = Instant::now();
        // 終局してから戻すまでの時間は対局時間に入れない
        if let Some(finished) = self.finished.take() {
            self.started += now - finished;
        }
        self.turn_started = now;
        true
    }
}

fn main() {
    inference::exit(run())
}

fn run() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = Config::load(&args.config)?;
    let model_path = args.model.unwrap_or_else(|| config.latest_model());
    let mut inference = config.inference;
    if args.cpu {
        inference.device = Device::Cpu;
    }
    let mut rules = Rules::default();
    for allowed in &args.allow {
        allowed.set(&mut rules, false);
    }
    let setup = Setup {
        model_path,
        inference,
        mcts: MctsConfig { simulations: args.simulations, dirichlet_epsilon: 0.0, ..config.selfplay.mcts() },
        mode: if args.free {
            Some(Mode::Free)
        } else {
            args.color.map(|c| match c {
                Color::Black => Mode::VsAi(Stone::Black),
                Color::White => Mode::VsAi(Stone::White),
            })
        },
        rules,
        candidates: args.candidates,
    };
    if args.plain || !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        plain::run(setup)
    } else {
        tui::run(setup)
    }
}
