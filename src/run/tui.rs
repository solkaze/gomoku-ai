//! 色付きの盤を描き、矢印キーかマウスで打つ対戦画面。
//!
//! モデルの読み込みと探索は別スレッドで行い、その間も画面 (進捗・経過時間) を更新し続ける。

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::bail;
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind,
};
use crossterm::style::{Color, Print, ResetColor, SetBackgroundColor, SetForegroundColor, Stylize};
use crossterm::terminal::{
    self, BeginSynchronizedUpdate, Clear, ClearType, DisableLineWrap, EnableLineWrap, EndSynchronizedUpdate,
    EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::{execute, queue};
use gomoku_engine::inference::LoadStage;
use gomoku_engine::{Board, CELLS, CENTER, GameResult, Rules, SIZE, SearchResult, Stone, pos};

use crate::{
    Engine, Forbidden, Game, Mode, RULES_WARNING, Setup, clock, coord, forbidden_names, forbidden_note, record,
    top_candidates,
};

type Rgb = (u8, u8, u8);

/// 盤の地の色 (マスの境目が分かるよう市松にする)
const WOOD: [Rgb; 2] = [(224, 186, 122), (208, 168, 102)];
const STAR: Rgb = (138, 104, 54);
const BLACK: Rgb = (22, 22, 22);
const WHITE: Rgb = (250, 250, 250);
const CURSOR: Rgb = (110, 205, 120);
/// 座標の入力中に、入れた x の列を塗る色 (マスの境目が分かるよう市松にする)
const COLUMN: [Rgb; 2] = [(150, 222, 150), (128, 206, 128)];
const WIN: Rgb = (245, 205, 50);
const HINT: Rgb = (60, 110, 195);
/// 直前の手と禁じ手の印
const MARK: Rgb = (225, 45, 35);
const GOOD: Rgb = (90, 200, 110);
const BAD: Rgb = (230, 90, 80);
const NOTE: Rgb = (235, 200, 80);
const BAR_REST: Rgb = (70, 70, 70);

const STARS: [usize; 5] = [pos(3, 3), pos(3, 11), CENTER, pos(11, 3), pos(11, 11)];
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const TITLE: &str = "五目並べ (連珠) ― AI と対戦";
const TITLE_FREE: &str = "五目並べ (連珠) ― 自由対局 (AI なし)";
/// 対局開始時の選択肢
const MODES: [Mode; 3] = [Mode::VsAi(Stone::Black), Mode::VsAi(Stone::White), Mode::Free];

/// 盤の左上 (目盛りの分だけ空ける)
const BOARD_X: u16 = 4;
const BOARD_Y: u16 = 2;
/// 盤の横や下に置く情報欄に確保する大きさ
const PANEL_W: u16 = 40;
const PANEL_H: u16 = 22;

/// 端末が 24bit 色に対応していなければ 256 色に丸める。
fn color((r, g, b): Rgb) -> Color {
    static TRUECOLOR: OnceLock<bool> = OnceLock::new();
    let truecolor =
        *TRUECOLOR.get_or_init(|| std::env::var("COLORTERM").is_ok_and(|v| v.contains("truecolor") || v.contains("24bit")));
    if truecolor {
        Color::Rgb { r, g, b }
    } else {
        let q = |v: u8| (v as u16 * 5 + 127) / 255;
        Color::AnsiValue((16 + 36 * q(r) + 6 * q(g) + q(b)) as u8)
    }
}

fn blend(a: Rgb, b: Rgb) -> Rgb {
    let mid = |x: u8, y: u8| ((x as u16 + y as u16) / 2) as u8;
    (mid(a.0, b.0), mid(a.1, b.1), mid(a.2, b.2))
}

fn stone_color(stone: Stone) -> Rgb {
    match stone {
        Stone::Black => BLACK,
        Stone::White => WHITE,
    }
}

fn stone_name(stone: Stone) -> &'static str {
    match stone {
        Stone::Black => "黒 (先手)",
        Stone::White => "白 (後手)",
    }
}

fn short_name(stone: Stone) -> &'static str {
    match stone {
        Stone::Black => "黒",
        Stone::White => "白",
    }
}

/// 文中に石を示すための、盤の地の色を付けた丸。
fn chip(stone: Stone) -> String {
    " ● ".with(color(stone_color(stone))).on(color(WOOD[0])).to_string()
}

/// 幅 width の横棒で percent を示す。
fn bar(percent: f64, width: usize, fill: Rgb) -> String {
    let filled = ((percent / 100.0 * width as f64).round() as usize).min(width);
    format!("{}{}", " ".repeat(filled).on(color(fill)), " ".repeat(width - filled).on(color(BAR_REST)))
}

fn put(out: &mut Vec<u8>, x: u16, y: u16, text: &str, fg: Rgb, bg: Rgb) -> io::Result<()> {
    queue!(out, MoveTo(x, y), SetForegroundColor(color(fg)), SetBackgroundColor(color(bg)), Print(text))
}

/// 勝ちが決まった石の並び (勝ちでなければ空)。
fn winning_line(board: &Board) -> Vec<usize> {
    let (GameResult::Win(winner), Some(last)) = (board.result(), board.last_move()) else {
        return Vec::new();
    };
    if board.get(last) != Some(winner) {
        return Vec::new();
    }
    let size = SIZE as isize;
    for (dr, dc) in [(0, 1), (1, 0), (1, 1), (1, -1)] {
        let mut line = vec![last];
        for sign in [-1, 1] {
            let (mut r, mut c) = ((last / SIZE) as isize + sign * dr, (last % SIZE) as isize + sign * dc);
            while (0..size).contains(&r) && (0..size).contains(&c) && board.get(pos(r as usize, c as usize)) == Some(winner) {
                line.push(pos(r as usize, c as usize));
                r += sign * dr;
                c += sign * dc;
            }
        }
        if line.len() >= 5 {
            return line;
        }
    }
    Vec::new()
}

/// 端末を対戦画面用に切り替え、破棄時 (パニック時も含む) に元へ戻す。
struct Terminal;

impl Terminal {
    fn enter() -> io::Result<Self> {
        terminal::enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen, Hide, DisableLineWrap, EnableMouseCapture)?;
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            Self::restore();
            hook(info);
        }));
        Ok(Self)
    }

    fn restore() {
        let _ = execute!(io::stdout(), DisableMouseCapture, EnableLineWrap, Show, LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        Self::restore();
    }
}

/// 探索スレッドからの通知
enum Reply {
    Stage(LoadStage),
    Ready,
    Failed(anyhow::Error),
    Searched(SearchResult),
}

/// モデルを読み込み、以後は送られてきた局面を探索して結果を返すスレッドを起動する。
/// stop が立つと探索を打ち切る。
fn spawn_engine(setup: Arc<Setup>, stop: Arc<AtomicBool>) -> (Sender<Board>, Receiver<Reply>, JoinHandle<()>) {
    let (request_tx, request_rx) = mpsc::channel::<Board>();
    let (reply_tx, reply_rx) = mpsc::channel();
    let handle = thread::spawn(move || {
        let loaded = Engine::load(&setup, &mut |stage| {
            let _ = reply_tx.send(Reply::Stage(stage));
        });
        let mut engine = match loaded {
            Ok(engine) => engine,
            Err(e) => {
                let _ = reply_tx.send(Reply::Failed(e));
                return;
            }
        };
        let _ = reply_tx.send(Reply::Ready);
        for board in request_rx {
            if reply_tx.send(Reply::Searched(engine.search_until(&board, &stop))).is_err() {
                break;
            }
        }
    });
    (request_tx, reply_rx, handle)
}

/// 盤と情報欄の配置
#[derive(Clone, Copy)]
struct Layout {
    /// 1 マスの幅と高さ (文字数)
    cw: u16,
    ch: u16,
    /// 情報欄の左上
    px: u16,
    py: u16,
}

impl Layout {
    /// 端末に収まる中で最も大きい配置を選ぶ。どれも収まらなければ None。
    fn fit(cols: u16, rows: u16) -> Option<Self> {
        let n = SIZE as u16;
        for (cw, ch) in [(4, 2), (2, 1)] {
            let (right, bottom) = (BOARD_X + n * cw, BOARD_Y + n * ch);
            if cols >= right + 3 + PANEL_W && rows > bottom {
                return Some(Self { cw, ch, px: right + 3, py: BOARD_Y });
            }
            if cols > right && rows > bottom + PANEL_H {
                return Some(Self { cw, ch, px: 1, py: bottom + 1 });
            }
        }
        None
    }

    fn large(&self) -> bool {
        self.ch == 2
    }

    /// 画面上の位置にあるマス
    fn cell_at(&self, col: u16, row: u16) -> Option<usize> {
        let (c, r) = (col.checked_sub(BOARD_X)? / self.cw, row.checked_sub(BOARD_Y)? / self.ch);
        (c < SIZE as u16 && r < SIZE as u16).then(|| pos(r as usize, c as usize))
    }
}

enum Screen {
    Loading,
    ChooseColor,
    Playing,
}

/// 探索スレッドに頼んでいること
enum Pending {
    AiMove,
    Hint,
}

/// 数字キーでの座標入力 (x y の順。x は左から、y は上から、どちらも 0〜14)。
/// x を入れるとその列を、y まで入れるとそのマスを盤上で示し、Enter で打つ (スペースは区切りにしか使わない)。
/// x と y はスペースで区切るが、x が 1 桁で決まるとき (0, 2〜9) と 2 桁入れたときは区切らずに y へ進む。
#[derive(Default)]
struct CoordInput {
    x: String,
    /// x を確定したら Some (y を 1 桁も入れていなければ空文字列)
    y: Option<String>,
}

impl CoordInput {
    fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    fn clear(&mut self) {
        *self = Self::default();
    }

    /// これ以上桁を足せないか (1 は 10〜14 になりうるので足せる)
    fn complete(digits: &str) -> bool {
        digits.len() >= 2 || (digits.len() == 1 && digits != "1")
    }

    fn valid(digits: &str) -> bool {
        digits.parse::<usize>().is_ok_and(|v| v < SIZE)
    }

    /// 数字を 1 つ足す。座標にならないときはエラーの説明を返す。
    fn push_digit(&mut self, d: char) -> Result<(), &'static str> {
        if self.y.is_none() {
            let next = format!("{}{d}", self.x);
            if !Self::complete(&self.x) && Self::valid(&next) {
                self.x = next;
                return Ok(());
            }
            if self.x.is_empty() {
                return Err("座標は 0〜14 です");
            }
            // x が決まっている (または 1 のあとに 5〜9 が来て 15 以上になる) なら、この数字から y
            self.y = Some(String::new());
        }
        let y = self.y.as_mut().expect("y に進んでいる");
        let next = format!("{y}{d}");
        if Self::complete(y) || !Self::valid(&next) {
            return Err("座標は 0〜14 です (Enter で打つ、Backspace で消す)");
        }
        *y = next;
        Ok(())
    }

    /// スペースで x を確定する。x がなければ何もせず false。
    fn separate(&mut self) -> bool {
        if self.x.is_empty() {
            return false;
        }
        self.y.get_or_insert_with(String::new);
        true
    }

    /// 最後に入れた 1 文字 (区切りを含む) を消す。
    fn backspace(&mut self) {
        match &mut self.y {
            Some(y) if !y.is_empty() => {
                y.pop();
            }
            Some(_) => self.y = None,
            None => {
                self.x.pop();
            }
        }
    }

    /// 入れた x (まだ 2 桁目が来うる場合も今の値)
    fn column(&self) -> Option<usize> {
        self.x.parse().ok()
    }

    /// x と y がそろったときのマス
    fn cell(&self) -> Option<usize> {
        let y: usize = self.y.as_deref()?.parse().ok()?;
        Some(pos(y, self.column()?))
    }

    /// 情報欄に出す入力の様子
    fn describe(&self) -> String {
        let y = self.y.as_deref().filter(|y| !y.is_empty()).unwrap_or("_");
        format!("x {}  y {y}", if self.x.is_empty() { "_" } else { &self.x })
    }
}

/// 探索結果のうち表示に使うもの
struct Analysis {
    /// 手番側から見た勝率 (%)
    win_rate: f64,
    candidates: Vec<(usize, f64)>,
}

impl Analysis {
    fn new(result: &SearchResult, candidates: usize) -> Self {
        // ほとんど読まれなかった手は候補として見せない
        let candidates = top_candidates(result, candidates).into_iter().filter(|&(_, pct)| pct >= 0.5).collect();
        Self { win_rate: (result.value as f64 + 1.0) * 50.0, candidates }
    }
}

struct App {
    setup: Arc<Setup>,
    /// 探索スレッドへの依頼 (AI を使わない起動では None)
    requests: Option<Sender<Board>>,
    screen: Screen,
    started: Instant,
    /// 読み込みで通った段階と、それぞれに入った時刻
    stages: Vec<(LoadStage, Instant)>,
    load_secs: f64,
    /// 対局の形式の選択でカーソルがある選択肢 (MODES の添字)
    choice: usize,
    /// 選択肢を描いた行 (クリックの判定用)
    choice_rows: [u16; MODES.len()],
    /// 黒に適用する禁じ手 (対局の形式の選択画面で切り替える。次の対局にも引き継ぐ)
    rules: Rules,
    /// 禁じ手の切り替えを描いた行 (クリックの判定用。Forbidden::ALL の順)
    rule_rows: [u16; Forbidden::ALL.len()],
    mode: Mode,
    game: Game,
    cursor: usize,
    /// 数字キーでの座標入力 (入力中はマウスと矢印キーを使わない)
    input: CoordInput,
    pending: Option<(Pending, Instant)>,
    /// AI の直前の着手 (着手、かかった秒数、読み)
    ai_move: Option<(usize, f64, Analysis)>,
    hint: Option<Analysis>,
    message: String,
    /// 終わった対局の棋譜
    records: Vec<String>,
    /// 対局の記録を書き出したファイル (終了時に表示する)
    saved: Vec<PathBuf>,
    layout: Option<Layout>,
    /// 次の描画で画面全体を消す
    clear: bool,
}

impl App {
    fn new(setup: Arc<Setup>, requests: Option<Sender<Board>>) -> Self {
        let rules = setup.rules;
        let mut app = Self {
            setup,
            requests,
            screen: Screen::Loading,
            started: Instant::now(),
            stages: Vec::new(),
            load_secs: 0.0,
            choice: 0,
            choice_rows: [0; MODES.len()],
            rules,
            rule_rows: [0; Forbidden::ALL.len()],
            mode: MODES[0],
            game: Game::new(rules),
            cursor: CENTER,
            input: CoordInput::default(),
            pending: None,
            ai_move: None,
            hint: None,
            message: String::new(),
            records: Vec::new(),
            saved: Vec::new(),
            layout: None,
            clear: true,
        };
        // 読み込むものがなければ、すぐ対局を始める
        if app.requests.is_none() {
            app.start_game(Mode::Free);
        }
        app
    }

    fn spinner(&self) -> &'static str {
        SPINNER[(self.started.elapsed().as_millis() / 80) as usize % SPINNER.len()]
    }

    fn show(&mut self, screen: Screen) {
        self.screen = screen;
        self.clear = true;
    }

    fn start_game(&mut self, mode: Mode) {
        self.mode = mode;
        self.game = Game::new(self.rules);
        self.cursor = CENTER;
        self.input.clear();
        self.ai_move = None;
        self.hint = None;
        self.message.clear();
        self.show(Screen::Playing);
    }

    fn new_game(&mut self) {
        if self.pending.is_some() {
            self.message = "AI の思考が終わるまでお待ちください".into();
            return;
        }
        if !self.game.moves.is_empty() {
            self.records.push(self.game.record());
        }
        self.save_record();
        match self.setup.mode {
            Some(mode) => self.start_game(mode),
            None => {
                self.game = Game::new(self.rules);
                self.show(Screen::ChooseColor);
            }
        }
    }

    /// 対局前に、禁じ手の有効・無効を切り替える。
    fn toggle_rule(&mut self, kind: Forbidden) {
        let on = kind.enabled(self.rules);
        kind.set(&mut self.rules, !on);
    }

    fn human_to_move(&self) -> bool {
        self.pending.is_none()
            && self.game.board.result() == GameResult::Ongoing
            && self.mode.is_human(self.game.board.side_to_move())
    }

    fn request(&mut self, pending: Pending) {
        // 探索スレッドが止まっていれば、次の受信で検出する
        if let Some(requests) = &self.requests {
            let _ = requests.send(self.game.board.clone());
        }
        self.pending = Some((pending, Instant::now()));
    }

    /// 人の入力を待たずに進められるところまで進める。
    fn advance(&mut self) {
        if !matches!(self.screen, Screen::Playing) || self.pending.is_some() {
            return;
        }
        if self.game.board.result() != GameResult::Ongoing {
            return;
        }
        if !self.mode.is_human(self.game.board.side_to_move()) {
            self.request(Pending::AiMove);
        } else if self.game.board.move_count() == 0 {
            self.game.play(CENTER).expect("天元は合法手");
            self.message = "黒の初手は天元 (7 7) に固定です".into();
        }
    }

    fn on_reply(&mut self, reply: Reply) -> anyhow::Result<()> {
        match reply {
            Reply::Stage(stage) => self.stages.push((stage, Instant::now())),
            Reply::Failed(e) => return Err(e),
            Reply::Ready => {
                self.load_secs = self.started.elapsed().as_secs_f64();
                match self.setup.mode {
                    Some(mode) => self.start_game(mode),
                    None => self.show(Screen::ChooseColor),
                }
            }
            Reply::Searched(result) => match self.pending.take() {
                Some((Pending::AiMove, start)) => {
                    let mv = result.best_move();
                    self.game.play_ai(mv, &result).expect("探索結果は合法手");
                    let analysis = Analysis::new(&result, self.setup.candidates);
                    self.ai_move = Some((mv, start.elapsed().as_secs_f64(), analysis));
                    self.hint = None;
                    self.message.clear();
                    self.after_move();
                }
                Some((Pending::Hint, _)) => self.hint = Some(Analysis::new(&result, self.setup.candidates.max(5))),
                None => {}
            },
        }
        Ok(())
    }

    fn place(&mut self, p: usize) {
        if self.pending.is_some() {
            self.message = "AI の思考が終わるまでお待ちください".into();
        } else if self.game.board.result() != GameResult::Ongoing {
            self.message = "対局は終了しています (n: 新しい対局)".into();
        } else if self.human_to_move() {
            match self.game.play(p) {
                Ok(_) => {
                    self.hint = None;
                    self.message.clear();
                    self.after_move();
                }
                Err(e) => self.message = format!("({}) には打てません: {e}", coord(p)),
            }
        }
    }

    /// 終局したら記録を書き出す。
    fn after_move(&mut self) {
        if self.game.board.result() != GameResult::Ongoing
            && let Some(path) = self.save_record()
        {
            self.message = format!("記録を書き出しました: {}", path.display());
        }
    }

    /// 今の対局の記録を書き出し、書き出したファイルを返す。失敗したら画面に知らせる。
    fn save_record(&mut self) -> Option<PathBuf> {
        match record::save(&self.setup, self.mode, &mut self.game) {
            Ok(Some(path)) => {
                if !self.saved.contains(&path) {
                    self.saved.push(path.clone());
                }
                Some(path)
            }
            Ok(None) => None,
            Err(e) => {
                self.message = format!("記録を書き出せませんでした: {e:#}");
                None
            }
        }
    }

    fn undo(&mut self) {
        if self.pending.is_some() {
            self.message = "AI の思考が終わるまでお待ちください".into();
        } else if self.game.undo(self.mode) {
            self.ai_move = None;
            self.hint = None;
            self.message = match self.mode {
                Mode::VsAi(_) => "あなたの直前の手まで戻しました".into(),
                Mode::Free => "1 手戻しました".into(),
            };
        } else {
            self.message = "戻せる手がありません".into();
        }
    }

    fn ask_hint(&mut self) {
        if self.mode == Mode::Free {
            self.message = "自由対局ではヒントは使えません".into();
        } else if self.human_to_move() {
            self.request(Pending::Hint);
        } else if self.pending.is_some() {
            self.message = "AI の思考が終わるまでお待ちください".into();
        }
    }

    /// 数字キーでの座標入力に 1 文字足す (数字かスペース)。
    fn type_coord(&mut self, key: char) {
        let result = if key == ' ' {
            // スペースは区切りだけ。y まで入れたあとに押しても打たない
            if self.input.cell().is_some() {
                Err("打つときは Enter を押してください")
            } else {
                self.input.separate();
                Ok(())
            }
        } else {
            self.input.push_digit(key)
        };
        match result {
            Ok(()) => self.message.clear(),
            Err(e) => self.message = e.into(),
        }
        if let Some(p) = self.input.cell() {
            self.cursor = p;
        }
    }

    /// 座標入力のマスに打つ。打てたら入力を消し、打てなければ (理由は place が知らせる) 入力を残す。
    fn place_input(&mut self) {
        let Some(p) = self.input.cell() else {
            self.message = "y も入力してください (x y の順)".into();
            return;
        };
        let before = self.game.moves.len();
        self.place(p);
        if self.game.moves.len() > before {
            self.input.clear();
        }
    }

    fn move_cursor(&mut self, dx: isize, dy: isize) {
        let step = |v: usize, d: isize| (v as isize + d).clamp(0, SIZE as isize - 1) as usize;
        self.cursor = pos(step(self.cursor / SIZE, dy), step(self.cursor % SIZE, dx));
    }

    /// 終了するなら true を返す。
    fn on_event(&mut self, event: Event) -> bool {
        match event {
            Event::Resize(..) => self.clear = true,
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                let ctrl_c = key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c');
                // 座標の入力中の Esc は入力の取り消し
                let entering = matches!(self.screen, Screen::Playing) && !self.input.is_empty();
                if ctrl_c || key.code == KeyCode::Char('q') || (key.code == KeyCode::Esc && !entering) {
                    return true;
                }
                match self.screen {
                    Screen::Loading => {}
                    Screen::ChooseColor => match key.code {
                        KeyCode::Up | KeyCode::Left => self.choice = (self.choice + MODES.len() - 1) % MODES.len(),
                        KeyCode::Down | KeyCode::Right | KeyCode::Tab => self.choice = (self.choice + 1) % MODES.len(),
                        KeyCode::Char('1') => self.start_game(MODES[0]),
                        KeyCode::Char('2') => self.start_game(MODES[1]),
                        KeyCode::Char('3') => self.start_game(MODES[2]),
                        KeyCode::Char('4') => self.toggle_rule(Forbidden::ALL[0]),
                        KeyCode::Char('5') => self.toggle_rule(Forbidden::ALL[1]),
                        KeyCode::Char('6') => self.toggle_rule(Forbidden::ALL[2]),
                        KeyCode::Enter | KeyCode::Char(' ') => self.start_game(MODES[self.choice]),
                        _ => {}
                    },
                    // 座標の入力中は数字・区切り・打つ・消す・取り消しだけを受け付ける
                    Screen::Playing if entering => match key.code {
                        KeyCode::Char(c @ '0'..='9') => self.type_coord(c),
                        KeyCode::Char(' ') => self.type_coord(' '),
                        KeyCode::Enter => self.place_input(),
                        KeyCode::Backspace => {
                            self.input.backspace();
                            self.message.clear();
                        }
                        KeyCode::Esc => {
                            self.input.clear();
                            self.message = "座標の入力を取り消しました".into();
                        }
                        KeyCode::Left | KeyCode::Right | KeyCode::Up | KeyCode::Down => {
                            self.message = "座標の入力中は矢印キーを使えません (Esc で取り消し)".into();
                        }
                        _ => {}
                    },
                    Screen::Playing => match key.code {
                        KeyCode::Char(c @ '0'..='9') => self.type_coord(c),
                        KeyCode::Left => self.move_cursor(-1, 0),
                        KeyCode::Right => self.move_cursor(1, 0),
                        KeyCode::Up => self.move_cursor(0, -1),
                        KeyCode::Down => self.move_cursor(0, 1),
                        // スペースは座標入力の区切りに使うので、打つのは Enter だけ
                        KeyCode::Enter => self.place(self.cursor),
                        KeyCode::Char('u') => self.undo(),
                        KeyCode::Char('h') => self.ask_hint(),
                        KeyCode::Char('n') => self.new_game(),
                        _ => {}
                    },
                }
            }
            Event::Mouse(mouse) => {
                let click = mouse.kind == MouseEventKind::Down(MouseButton::Left);
                match self.screen {
                    Screen::Loading => {}
                    Screen::ChooseColor if click => {
                        if let Some(i) = self.choice_rows.iter().position(|&row| row == mouse.row) {
                            self.start_game(MODES[i]);
                        } else if let Some(i) = self.rule_rows.iter().position(|&row| row == mouse.row) {
                            self.toggle_rule(Forbidden::ALL[i]);
                        }
                    }
                    Screen::ChooseColor => {}
                    // 座標の入力中は、誤クリックで打たないようマウスを使わない
                    Screen::Playing if !self.input.is_empty() => {
                        if click {
                            self.message = "座標の入力中はクリックで打てません (Esc で取り消し)".into();
                        }
                    }
                    Screen::Playing => {
                        if let Some(p) = self.layout.and_then(|lay| lay.cell_at(mouse.column, mouse.row)) {
                            self.cursor = p;
                            if click {
                                self.place(p);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        false
    }

    fn draw(&mut self) -> io::Result<()> {
        let mut out = Vec::new();
        queue!(out, BeginSynchronizedUpdate, ResetColor)?;
        if std::mem::take(&mut self.clear) {
            queue!(out, Clear(ClearType::All))?;
        }
        let (cols, rows) = terminal::size()?;
        match self.screen {
            Screen::Loading => {
                let lines = self.loading_lines();
                draw_lines(&mut out, 2, 1, rows, &lines)?;
            }
            Screen::ChooseColor => {
                let (lines, choice_rows, rule_rows) = self.choose_lines();
                self.choice_rows = choice_rows.map(|i| 1 + i as u16);
                self.rule_rows = rule_rows.map(|i| 1 + i as u16);
                draw_lines(&mut out, 2, 1, rows, &lines)?;
            }
            Screen::Playing => {
                self.layout = Layout::fit(cols, rows);
                match self.layout {
                    Some(lay) => {
                        match self.mode {
                            Mode::VsAi(_) => queue!(
                                out,
                                MoveTo(1, 0),
                                Print(TITLE.bold()),
                                Print(format!("   {}", self.setup.describe()).dark_grey())
                            )?,
                            Mode::Free => queue!(out, MoveTo(1, 0), Print(TITLE_FREE.bold()))?,
                        }
                        self.draw_board(&mut out, &lay)?;
                        draw_lines(&mut out, lay.px, lay.py, rows, &self.panel_lines(&lay))?;
                    }
                    None => {
                        let lines = ["端末が小さすぎます。".to_string(), "ウィンドウを広げてください (横 77 x 縦 18 文字以上)。".to_string()];
                        queue!(out, Clear(ClearType::All))?;
                        draw_lines(&mut out, 0, 0, rows, &lines)?;
                        self.clear = true;
                    }
                }
            }
        }
        queue!(out, ResetColor, EndSynchronizedUpdate)?;
        let mut stdout = io::stdout().lock();
        stdout.write_all(&out)?;
        stdout.flush()
    }

    fn loading_lines(&self) -> Vec<String> {
        let all = LoadStage::stages(self.setup.inference.device);
        let now = Instant::now();
        let mut lines =
            vec![TITLE.bold().to_string(), String::new(), format!("モデル: {}", self.setup.describe()), String::new()];
        for &stage in all {
            let line = match self.stages.iter().position(|&(s, _)| s == stage) {
                // 次の段階に入っていれば完了している
                Some(i) => match self.stages.get(i + 1) {
                    Some(&(_, end)) => format!(
                        "{} {}  {}",
                        "✔".with(color(GOOD)),
                        stage.label(),
                        format!("{:.1} 秒", (end - self.stages[i].1).as_secs_f64()).dark_grey()
                    ),
                    None => format!(
                        "{} {}  {:.1} 秒",
                        self.spinner().with(color(NOTE)),
                        stage.label().bold(),
                        (now - self.stages[i].1).as_secs_f64()
                    ),
                },
                None => format!("· {}", stage.label()).dark_grey().to_string(),
            };
            lines.push(line);
        }
        let done = self.stages.len().saturating_sub(1);
        lines.push(String::new());
        lines.push(format!(
            "{}  {done}/{}   読み込み中… {:.1} 秒",
            bar(done as f64 * 100.0 / all.len() as f64, 32, GOOD),
            all.len(),
            self.started.elapsed().as_secs_f64()
        ));
        lines.push(String::new());
        lines.push("q 終了".dark_grey().to_string());
        lines
    }

    /// 対局の形式の選択画面の行と、そのうち選択肢 (MODES の順) と禁じ手の切り替え (Forbidden::ALL の順) の行番号。
    fn choose_lines(&self) -> (Vec<String>, [usize; MODES.len()], [usize; Forbidden::ALL.len()]) {
        let mut lines = vec![
            TITLE.bold().to_string(),
            String::new(),
            format!("モデル: {}", self.setup.describe()),
            format!("{} 読み込み完了 ({:.1} 秒)", "✔".with(color(GOOD)), self.load_secs),
            String::new(),
            "先手・後手、または自由対局を選んでください".to_string(),
            String::new(),
        ];
        let first = lines.len();
        let black_note = forbidden_note(self.rules);
        for (i, mode) in MODES.into_iter().enumerate() {
            let (name, note) = match mode {
                Mode::VsAi(Stone::Black) => (chip(Stone::Black) + " 黒 (先手)", black_note.as_str()),
                Mode::VsAi(Stone::White) => (chip(Stone::White) + " 白 (後手)", ""),
                Mode::Free => (chip(Stone::Black) + &chip(Stone::White) + " 自由対局", "AI を使わず、黒と白の両方を自分で打つ"),
            };
            let label = format!("{}  {name}", i + 1);
            lines.push(if i == self.choice {
                format!(" {} {}   {}", "▶".with(color(CURSOR)), label.bold(), note.dark_grey())
            } else {
                format!("   {label}   {}", note.dark_grey())
            });
            lines.push(String::new());
        }
        lines.push("黒の禁じ手 (4〜6 キーかクリックで有効・無効を切り替え)".to_string());
        let first_rule = lines.len();
        for (i, kind) in Forbidden::ALL.into_iter().enumerate() {
            let state = if kind.enabled(self.rules) {
                "有効".with(color(GOOD)).bold().to_string()
            } else {
                "無効".with(color(BAD)).bold().to_string()
            };
            let note = match (kind, kind.enabled(self.rules)) {
                (Forbidden::Overline, false) => "黒も六以上の連で勝ち",
                (_, false) => "黒も打てる",
                (_, true) => "",
            };
            lines.push(format!("   {}  {}  {state}   {}", i + MODES.len() + 1, kind.name(), note.dark_grey()));
        }
        // 行数が変わると選択肢の位置がずれるので、注意がないときも 1 行空ける
        let warning = if self.rules == Rules::default() { "" } else { RULES_WARNING };
        lines.push(format!("   {}", warning.with(color(NOTE))));
        lines.push(String::new());
        lines.push("↑↓ で選んで Enter (1〜3 キーやクリックでも選べます)   q 終了".dark_grey().to_string());
        (lines, std::array::from_fn(|i| first + 2 * i), std::array::from_fn(|i| first_rule + i))
    }

    fn panel_lines(&self, lay: &Layout) -> Vec<String> {
        let board = &self.game.board;
        let game = &self.game;
        // 時間を表示する順 (人、AI。自由対局では黒、白) と、それぞれの呼び名
        let sides = match self.mode {
            Mode::VsAi(human) => [(human, "あなた"), (human.opponent(), "AI")],
            Mode::Free => [(Stone::Black, "黒"), (Stone::White, "白")],
        };
        let mut lines = vec![
            match self.mode {
                Mode::VsAi(human) => {
                    let ai = human.opponent();
                    format!("あなた {} {}   AI {} {}", chip(human), stone_name(human), chip(ai), stone_name(ai))
                }
                Mode::Free => format!("{} 黒と {} 白を交互に自分で打つ", chip(Stone::Black), chip(Stone::White)),
            },
            format!(
                "対局時間 {}   {} {}   {} {}",
                clock(game.elapsed()).bold(),
                sides[0].1,
                clock(game.total(sides[0].0)),
                sides[1].1,
                clock(game.total(sides[1].0))
            ),
            String::new(),
        ];
        lines.push(match (board.result(), &self.pending) {
            (GameResult::Win(winner), _) => {
                let note = if board.lost_by_forbidden() { "黒の禁じ手・" } else { "" };
                let text = match self.mode {
                    Mode::VsAi(human) if winner == human => "あなたの勝ちです".to_string().with(color(GOOD)).bold(),
                    Mode::VsAi(_) => "AI の勝ちです".to_string().with(color(BAD)).bold(),
                    Mode::Free => format!("{}の勝ちです", short_name(winner)).with(color(GOOD)).bold(),
                };
                format!("{text} ({note}{} 手)", board.move_count())
            }
            (GameResult::Draw, _) => "引き分けです".bold().to_string(),
            (GameResult::Ongoing, Some((pending, start))) => format!(
                "{} {} {:.1} 秒",
                self.spinner().with(color(NOTE)),
                match pending {
                    Pending::AiMove => "AI が考えています…",
                    Pending::Hint => "ヒントを計算しています…",
                },
                start.elapsed().as_secs_f64()
            ),
            (GameResult::Ongoing, None) => {
                let turn = match self.mode {
                    Mode::VsAi(_) => "あなたの番です".bold().to_string(),
                    Mode::Free => {
                        let side = board.side_to_move();
                        format!("{} {}", chip(side), format!("{}の番です", short_name(side)).bold())
                    }
                };
                format!("{turn} {} 手目  {:.1} 秒", board.move_count() + 1, game.thinking().as_secs_f64())
            }
        });
        lines.push(self.message.as_str().with(color(NOTE)).to_string());
        lines.push(String::new());

        let candidate_lines = |candidates: &[(usize, f64)]| -> Vec<String> {
            candidates
                .chunks(3)
                .map(|chunk| {
                    let moves: Vec<String> = chunk.iter().map(|&(mv, pct)| format!("({}) {pct:.0}%", coord(mv))).collect();
                    format!("  {}", moves.join("   "))
                })
                .collect()
        };
        // 人の直前の手とかかった時間 (天元に固定の初手は除く)
        let human_moves: Vec<String> = sides
            .iter()
            .filter(|&&(stone, _)| self.mode.is_human(stone))
            .filter_map(|&(stone, name)| {
                let (i, mv, time) = game.last_move_by(stone).filter(|&(i, ..)| i > 0)?;
                let label = format!("{name}の着手").bold();
                let time = format!("{:.1} 秒 ({} 手目)", time.as_secs_f64(), i + 1).dark_grey();
                Some(format!("{label} ({})  {time}", coord(mv)))
            })
            .collect();
        if !human_moves.is_empty() {
            lines.extend(human_moves);
            lines.push(String::new());
        }
        if let Some((mv, secs, analysis)) = &self.ai_move {
            lines.push(format!("{} ({})  {}", "AI の着手".bold(), coord(*mv), format!("{secs:.1} 秒").dark_grey()));
            lines.push(format!("  {}  AI の勝率見込み {:.0}%", bar(analysis.win_rate, 16, BAD), analysis.win_rate));
            if !analysis.candidates.is_empty() {
                lines.push("  AI の候補手 (x y) と探索の割合".dark_grey().to_string());
                lines.extend(candidate_lines(&analysis.candidates));
            }
            lines.push(String::new());
        }
        if let Some(hint) = &self.hint {
            lines.push(format!("{}  盤上の{}が候補手", "ヒント".bold(), "青いマス".with(color(WHITE)).on(color(HINT))));
            lines.push(format!("  {}  あなたの勝率見込み {:.0}%", bar(hint.win_rate, 16, GOOD), hint.win_rate));
            lines.push(String::new());
        }

        let last_mark = if lay.large() { "赤い点" } else { "赤いマス" };
        let (cursor_line, place_line) = if self.input.is_empty() {
            (
                format!("カーソル ({})   ←↑↓→・マウス: 移動", coord(self.cursor)),
                "Enter・クリック: 打つ   数字: 座標を入力 (x y)".to_string(),
            )
        } else {
            (
                format!("{} {}", "座標入力".with(color(CURSOR)).bold(), self.input.describe().bold()),
                "Enter: 打つ   Space: 区切り   Backspace: 消す   Esc: 取り消し".to_string(),
            )
        };
        let keys = [
            cursor_line,
            place_line,
            match self.mode {
                Mode::VsAi(_) => "u: 待った   h: ヒント".to_string(),
                Mode::Free => "u: 1 手戻す".to_string(),
            },
            "n: 新しい対局   q: 終了".to_string(),
            format!(
                "{last_mark}: 直前の手   {}: 禁じ手 ({})",
                "×".with(color(MARK)),
                forbidden_names(board.rules()).as_deref().unwrap_or("なし")
            ),
        ];
        lines.extend(keys.into_iter().map(|line| line.dark_grey().to_string()));
        lines
    }

    fn draw_board(&self, out: &mut Vec<u8>, lay: &Layout) -> io::Result<()> {
        let board = &self.game.board;
        let human_to_move = self.human_to_move();
        let forbidden = (human_to_move && board.side_to_move() == Stone::Black).then(|| board.forbidden_mask());
        let win = winning_line(board);
        let hints: &[(usize, f64)] = self.hint.as_ref().map_or(&[], |hint| &hint.candidates);

        // 座標の入力中: x だけなら列を示し、y までそろえばカーソルをそのマスに置く (type_coord)
        let input_column = if self.input.is_empty() { None } else { self.input.column() };
        let column_only = input_column.filter(|_| self.input.cell().is_none());
        let show_cursor = self.input.is_empty() || self.input.cell().is_some();
        let (cursor_col, cursor_row) = match column_only {
            Some(x) => (Some(x), None),
            None if show_cursor => (Some(self.cursor % SIZE), Some(self.cursor / SIZE)),
            None => (None, None),
        };

        // 目盛り。カーソル (または入力中の列) のある行と列を目立たせる
        for i in 0..SIZE {
            let label = |text: String, on: bool| if on { text.with(color(CURSOR)).bold() } else { text.dark_grey() };
            let n = i as u16;
            let column = format!("{i:^width$}", width = lay.cw as usize);
            queue!(out, MoveTo(BOARD_X + n * lay.cw, BOARD_Y - 1), Print(label(column, cursor_col == Some(i))))?;
            queue!(out, MoveTo(BOARD_X - 3, BOARD_Y + n * lay.ch), Print(label(format!("{i:>2}"), cursor_row == Some(i))))?;
        }

        for p in 0..CELLS {
            let (r, c) = (p / SIZE, p % SIZE);
            let (x, y) = (BOARD_X + c as u16 * lay.cw, BOARD_Y + r as u16 * lay.ch);
            let stone = board.get(p);
            let is_last = board.last_move() == Some(p);
            let is_cursor = show_cursor && self.cursor == p;
            let in_column = column_only == Some(c);
            let is_forbidden = forbidden.as_ref().is_some_and(|f| f[p]);
            let hint = hints.iter().position(|&(mv, _)| mv == p).filter(|_| stone.is_none());
            let bg = if win.contains(&p) {
                WIN
            } else if is_cursor {
                CURSOR
            } else if hint.is_some() {
                HINT
            } else if in_column {
                COLUMN[(r + c) % 2]
            } else {
                WOOD[(r + c) % 2]
            };
            // カーソルのある空点には、打てる場合に限り手番の石を薄く見せる
            let ghost = is_cursor && human_to_move && stone.is_none() && !is_forbidden && hint.is_none();
            let disc = match stone {
                Some(stone) => Some(stone_color(stone)),
                None if ghost => Some(blend(stone_color(board.side_to_move()), bg)),
                None => None,
            };

            if lay.large() {
                let (top, bottom, fg) = match (disc, hint) {
                    (Some(fg), _) => ("▟██▙".to_string(), "▜██▛".to_string(), fg),
                    (None, _) if is_forbidden => (" ▚▞ ".to_string(), " ▞▚ ".to_string(), MARK),
                    (None, Some(rank)) => (format!(" #{:<2}", rank + 1), format!("{:>3.0}%", hints[rank].1), WHITE),
                    (None, None) if STARS.contains(&p) => (" ▗▖ ".to_string(), " ▝▘ ".to_string(), STAR),
                    (None, None) => ("    ".to_string(), "    ".to_string(), bg),
                };
                put(out, x, y, &top, fg, bg)?;
                put(out, x, y + 1, &bottom, fg, bg)?;
                if let (true, Some(stone)) = (is_last, stone) {
                    put(out, x + 1, y, "▗▖", MARK, stone_color(stone))?;
                    put(out, x + 1, y + 1, "▝▘", MARK, stone_color(stone))?;
                }
            } else {
                let bg = if is_last && !is_cursor && win.is_empty() { blend(MARK, WOOD[0]) } else { bg };
                let (glyph, fg) = match (disc, hint) {
                    (Some(fg), _) => ("●".to_string(), fg),
                    (None, _) if is_forbidden => ("x".to_string(), MARK),
                    (None, Some(rank)) => ((rank + 1).to_string(), WHITE),
                    (None, None) if STARS.contains(&p) => ("+".to_string(), STAR),
                    (None, None) => (" ".to_string(), bg),
                };
                // ● の幅は端末によって違うので、先にマスを塗ってから置く
                put(out, x, y, "  ", bg, bg)?;
                put(out, x, y, &glyph, fg, bg)?;
            }
        }
        queue!(out, ResetColor)
    }

    fn event_loop(&mut self, replies: Option<&Receiver<Reply>>) -> anyhow::Result<()> {
        loop {
            while let Some(replies) = replies {
                match replies.try_recv() {
                    Ok(reply) => self.on_reply(reply)?,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => bail!("探索スレッドが異常終了しました"),
                }
            }
            self.advance();
            self.draw()?;

            // 読み込み中・探索中・対局中は、経過時間を表示し続けるために短い間隔で描き直す
            let animating = match self.screen {
                Screen::Loading => true,
                Screen::ChooseColor => false,
                Screen::Playing => self.pending.is_some() || self.game.board.result() == GameResult::Ongoing,
            };
            if event::poll(Duration::from_millis(if animating { 80 } else { 1000 }))? {
                loop {
                    if self.on_event(event::read()?) {
                        return Ok(());
                    }
                    if !event::poll(Duration::ZERO)? {
                        break;
                    }
                }
            }
        }
    }
}

/// (x, y) から下へ 1 行ずつ描き、残りの行は画面の下端まで消す。
fn draw_lines(out: &mut Vec<u8>, x: u16, y: u16, rows: u16, lines: &[String]) -> io::Result<()> {
    for row in y..rows {
        let line = lines.get((row - y) as usize).map_or("", |line| line.as_str());
        queue!(out, MoveTo(x, row), Print(line), Clear(ClearType::UntilNewLine))?;
    }
    Ok(())
}

pub fn run(setup: Setup) -> anyhow::Result<()> {
    let setup = Arc::new(setup);
    let stop = Arc::new(AtomicBool::new(false));
    // 自由対局だけなら AI は要らないので、モデルを読み込まない
    let (requests, replies, engine) = if setup.mode == Some(Mode::Free) {
        (None, None, None)
    } else {
        let (requests, replies, engine) = spawn_engine(Arc::clone(&setup), Arc::clone(&stop));
        (Some(requests), Some(replies), Some(engine))
    };
    let mut app = App::new(setup, requests);
    let terminal = Terminal::enter()?;
    let result = app.event_loop(replies.as_ref());
    drop(terminal);

    // 推論の最中にプロセスを終えると onnxruntime の後始末と競合して落ちるので、探索スレッドの終了を待つ
    stop.store(true, Ordering::Relaxed);
    if !app.game.moves.is_empty() {
        let record = app.game.record();
        app.records.push(record);
    }
    // 終局していない対局も、途中までを記録に残す
    app.save_record();
    let App { requests, records, saved, message, .. } = app;
    drop(requests);
    if let Some(engine) = engine {
        let _ = engine.join();
    }

    for record in &records {
        println!("{record}");
    }
    for path in &saved {
        println!("記録: {}", path.display());
    }
    if saved.is_empty() && message.starts_with("記録を書き出せませんでした") {
        println!("{message}");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::CoordInput;
    use gomoku_engine::pos;

    fn typed(keys: &str) -> CoordInput {
        let mut input = CoordInput::default();
        for k in keys.chars() {
            if k == ' ' {
                input.separate();
            } else {
                let _ = input.push_digit(k);
            }
        }
        input
    }

    #[test]
    fn x_then_y_with_or_without_a_space() {
        // x を入れた段階では列だけ
        let input = typed("7");
        assert_eq!((input.column(), input.cell()), (Some(7), None));
        // 1 桁で決まる x はスペースなしで y に進む
        assert_eq!(typed("77").cell(), Some(pos(7, 7)));
        assert_eq!(typed("7 7").cell(), Some(pos(7, 7)));
        // x は左から、y は上から
        assert_eq!(typed("3 12").cell(), Some(pos(12, 3)));
    }

    #[test]
    fn one_waits_for_a_second_digit() {
        // 1 のあとは 10〜14 になりうるので、まだ y に進まない
        assert_eq!(typed("1").cell(), None);
        assert_eq!(typed("12").column(), Some(12));
        assert_eq!(typed("12").cell(), None);
        assert_eq!(typed("123").cell(), Some(pos(3, 12)));
        // 15 以上にはならないので、1 のあとの 5〜9 は y
        assert_eq!(typed("15").cell(), Some(pos(5, 1)));
        // スペースで x = 1 に決められる
        assert_eq!(typed("1 2").cell(), Some(pos(2, 1)));
        assert_eq!(typed("1 14").cell(), Some(pos(14, 1)));
    }

    #[test]
    fn out_of_range_digits_are_rejected() {
        let mut input = typed("7 7");
        assert!(input.push_digit('7').is_err());
        assert_eq!(input.cell(), Some(pos(7, 7)));
        let mut input = typed("7 1");
        assert!(input.push_digit('5').is_err());
        assert!(input.push_digit('4').is_ok());
        assert_eq!(input.cell(), Some(pos(14, 7)));
    }

    #[test]
    fn backspace_steps_back_through_the_separator() {
        let mut input = typed("7 12");
        input.backspace();
        assert_eq!(input.cell(), Some(pos(1, 7)));
        input.backspace();
        assert_eq!((input.column(), input.cell()), (Some(7), None));
        input.backspace(); // 区切りを消す
        input.backspace();
        assert!(input.is_empty());
        assert_eq!(input.describe(), "x _  y _");
    }
}
