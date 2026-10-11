//! 色付きの盤を描き、矢印キーかマウスで打つ対戦画面。
//!
//! モデルの読み込みと探索は別スレッドで行い、その間も画面 (進捗・経過時間) を更新し続ける。

use std::io::{self, Write};
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
use gomoku_engine::{Board, CELLS, CENTER, GameResult, SIZE, SearchResult, Stone, pos};

use crate::{Engine, Game, Setup, coord, top_candidates};

type Rgb = (u8, u8, u8);

/// 盤の地の色 (マスの境目が分かるよう市松にする)
const WOOD: [Rgb; 2] = [(224, 186, 122), (208, 168, 102)];
const STAR: Rgb = (138, 104, 54);
const BLACK: Rgb = (22, 22, 22);
const WHITE: Rgb = (250, 250, 250);
const CURSOR: Rgb = (110, 205, 120);
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

/// 盤の左上 (目盛りの分だけ空ける)
const BOARD_X: u16 = 4;
const BOARD_Y: u16 = 2;
/// 盤の横や下に置く情報欄に確保する大きさ
const PANEL_W: u16 = 40;
const PANEL_H: u16 = 20;

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
    requests: Sender<Board>,
    screen: Screen,
    started: Instant,
    /// 読み込みで通った段階と、それぞれに入った時刻
    stages: Vec<(LoadStage, Instant)>,
    load_secs: f64,
    /// 先手・後手の選択でカーソルがある側
    choice: Stone,
    /// 先手・後手の選択肢を描いた行 (クリックの判定用)
    choice_rows: [u16; 2],
    human: Stone,
    game: Game,
    cursor: usize,
    pending: Option<(Pending, Instant)>,
    /// AI の直前の着手 (着手、かかった秒数、読み)
    ai_move: Option<(usize, f64, Analysis)>,
    hint: Option<Analysis>,
    message: String,
    /// 終わった対局の棋譜
    records: Vec<String>,
    layout: Option<Layout>,
    /// 次の描画で画面全体を消す
    clear: bool,
}

impl App {
    fn new(setup: Arc<Setup>, requests: Sender<Board>) -> Self {
        Self {
            setup,
            requests,
            screen: Screen::Loading,
            started: Instant::now(),
            stages: Vec::new(),
            load_secs: 0.0,
            choice: Stone::Black,
            choice_rows: [0; 2],
            human: Stone::Black,
            game: Game::new(),
            cursor: CENTER,
            pending: None,
            ai_move: None,
            hint: None,
            message: String::new(),
            records: Vec::new(),
            layout: None,
            clear: true,
        }
    }

    fn spinner(&self) -> &'static str {
        SPINNER[(self.started.elapsed().as_millis() / 80) as usize % SPINNER.len()]
    }

    fn show(&mut self, screen: Screen) {
        self.screen = screen;
        self.clear = true;
    }

    fn start_game(&mut self, human: Stone) {
        self.human = human;
        self.game = Game::new();
        self.cursor = CENTER;
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
        match self.setup.color {
            Some(human) => self.start_game(human),
            None => {
                self.game = Game::new();
                self.show(Screen::ChooseColor);
            }
        }
    }

    fn human_to_move(&self) -> bool {
        self.pending.is_none()
            && self.game.board.result() == GameResult::Ongoing
            && self.game.board.side_to_move() == self.human
    }

    fn request(&mut self, pending: Pending) {
        // 探索スレッドが止まっていれば、次の受信で検出する
        let _ = self.requests.send(self.game.board.clone());
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
        if self.game.board.side_to_move() != self.human {
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
                match self.setup.color {
                    Some(human) => self.start_game(human),
                    None => self.show(Screen::ChooseColor),
                }
            }
            Reply::Searched(result) => match self.pending.take() {
                Some((Pending::AiMove, start)) => {
                    let mv = result.best_move();
                    self.game.play(mv).expect("探索結果は合法手");
                    let analysis = Analysis::new(&result, self.setup.candidates);
                    self.ai_move = Some((mv, start.elapsed().as_secs_f64(), analysis));
                    self.hint = None;
                    self.message.clear();
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
                }
                Err(e) => self.message = format!("({}) には打てません: {e}", coord(p)),
            }
        }
    }

    fn undo(&mut self) {
        if self.pending.is_some() {
            self.message = "AI の思考が終わるまでお待ちください".into();
        } else if self.game.undo(self.human) {
            self.ai_move = None;
            self.hint = None;
            self.message = "あなたの直前の手まで戻しました".into();
        } else {
            self.message = "戻せる手がありません".into();
        }
    }

    fn ask_hint(&mut self) {
        if self.human_to_move() {
            self.request(Pending::Hint);
        } else if self.pending.is_some() {
            self.message = "AI の思考が終わるまでお待ちください".into();
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
                if ctrl_c || matches!(key.code, KeyCode::Char('q') | KeyCode::Esc) {
                    return true;
                }
                match self.screen {
                    Screen::Loading => {}
                    Screen::ChooseColor => match key.code {
                        KeyCode::Up | KeyCode::Down | KeyCode::Left | KeyCode::Right | KeyCode::Tab => {
                            self.choice = self.choice.opponent()
                        }
                        KeyCode::Char('1') => self.start_game(Stone::Black),
                        KeyCode::Char('2') => self.start_game(Stone::White),
                        KeyCode::Enter | KeyCode::Char(' ') => self.start_game(self.choice),
                        _ => {}
                    },
                    Screen::Playing => match key.code {
                        KeyCode::Left => self.move_cursor(-1, 0),
                        KeyCode::Right => self.move_cursor(1, 0),
                        KeyCode::Up => self.move_cursor(0, -1),
                        KeyCode::Down => self.move_cursor(0, 1),
                        KeyCode::Enter | KeyCode::Char(' ') => self.place(self.cursor),
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
                            self.start_game([Stone::Black, Stone::White][i]);
                        }
                    }
                    Screen::ChooseColor => {}
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
                let (lines, choice_rows) = self.choose_lines();
                self.choice_rows = choice_rows.map(|i| 1 + i as u16);
                draw_lines(&mut out, 2, 1, rows, &lines)?;
            }
            Screen::Playing => {
                self.layout = Layout::fit(cols, rows);
                match self.layout {
                    Some(lay) => {
                        queue!(out, MoveTo(1, 0), Print(TITLE.bold()), Print(format!("   {}", self.setup.describe()).dark_grey()))?;
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

    /// 先手・後手の選択画面の行と、そのうち選択肢 (先手、後手) の行番号。
    fn choose_lines(&self) -> (Vec<String>, [usize; 2]) {
        let mut lines = vec![
            TITLE.bold().to_string(),
            String::new(),
            format!("モデル: {}", self.setup.describe()),
            format!("{} 読み込み完了 ({:.1} 秒)", "✔".with(color(GOOD)), self.load_secs),
            String::new(),
            "先手・後手を選んでください".to_string(),
            String::new(),
        ];
        let first = lines.len();
        for (key, stone, note) in [(1, Stone::Black, "三三・四四・長連は禁じ手 (打つと負け)"), (2, Stone::White, "")] {
            let label = format!("{key}  {} {}", chip(stone), stone_name(stone));
            lines.push(if stone == self.choice {
                format!(" {} {}   {}", "▶".with(color(CURSOR)), label.bold(), note.dark_grey())
            } else {
                format!("   {label}   {}", note.dark_grey())
            });
            lines.push(String::new());
        }
        lines.push("↑↓ で選んで Enter (1 / 2 キーやクリックでも選べます)   q 終了".dark_grey().to_string());
        (lines, [first, first + 2])
    }

    fn panel_lines(&self, lay: &Layout) -> Vec<String> {
        let board = &self.game.board;
        let ai = self.human.opponent();
        let mut lines = vec![
            format!("あなた {} {}   AI {} {}", chip(self.human), stone_name(self.human), chip(ai), stone_name(ai)),
            String::new(),
        ];
        lines.push(match (board.result(), &self.pending) {
            (GameResult::Win(winner), _) => {
                let note = if board.lost_by_forbidden() { "黒の禁じ手・" } else { "" };
                let text = if winner == self.human {
                    "あなたの勝ちです".with(color(GOOD)).bold()
                } else {
                    "AI の勝ちです".with(color(BAD)).bold()
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
            (GameResult::Ongoing, None) => format!(
                "{} {} 手目   カーソル ({})",
                "あなたの番です".bold(),
                board.move_count() + 1,
                coord(self.cursor)
            ),
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
        let keys = [
            "←↑↓→・マウス: カーソル移動".to_string(),
            "Enter・クリック: 打つ".to_string(),
            "u: 待った   h: ヒント".to_string(),
            "n: 新しい対局   q: 終了".to_string(),
            format!("{last_mark}: 直前の手   {}: 黒の禁じ手 (打つと負け)", "×".with(color(MARK))),
        ];
        lines.extend(keys.into_iter().map(|line| line.dark_grey().to_string()));
        lines
    }

    fn draw_board(&self, out: &mut Vec<u8>, lay: &Layout) -> io::Result<()> {
        let board = &self.game.board;
        let human_to_move = self.human_to_move();
        let forbidden = (human_to_move && self.human == Stone::Black).then(|| board.forbidden_mask());
        let win = winning_line(board);
        let hints: &[(usize, f64)] = self.hint.as_ref().map_or(&[], |hint| &hint.candidates);

        // 目盛り。カーソルのある行と列を目立たせる
        for i in 0..SIZE {
            let label = |text: String, on: bool| if on { text.with(color(CURSOR)).bold() } else { text.dark_grey() };
            let n = i as u16;
            let column = format!("{i:^width$}", width = lay.cw as usize);
            queue!(out, MoveTo(BOARD_X + n * lay.cw, BOARD_Y - 1), Print(label(column, self.cursor % SIZE == i)))?;
            queue!(out, MoveTo(BOARD_X - 3, BOARD_Y + n * lay.ch), Print(label(format!("{i:>2}"), self.cursor / SIZE == i)))?;
        }

        for p in 0..CELLS {
            let (r, c) = (p / SIZE, p % SIZE);
            let (x, y) = (BOARD_X + c as u16 * lay.cw, BOARD_Y + r as u16 * lay.ch);
            let stone = board.get(p);
            let is_last = board.last_move() == Some(p);
            let is_cursor = self.cursor == p;
            let is_forbidden = forbidden.as_ref().is_some_and(|f| f[p]);
            let hint = hints.iter().position(|&(mv, _)| mv == p).filter(|_| stone.is_none());
            let bg = if win.contains(&p) {
                WIN
            } else if is_cursor {
                CURSOR
            } else if hint.is_some() {
                HINT
            } else {
                WOOD[(r + c) % 2]
            };
            // カーソルのある空点には、打てる場合に限り自分の石を薄く見せる
            let ghost = is_cursor && human_to_move && stone.is_none() && !is_forbidden && hint.is_none();
            let disc = match stone {
                Some(stone) => Some(stone_color(stone)),
                None if ghost => Some(blend(stone_color(self.human), bg)),
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

    fn event_loop(&mut self, replies: &Receiver<Reply>) -> anyhow::Result<()> {
        loop {
            loop {
                match replies.try_recv() {
                    Ok(reply) => self.on_reply(reply)?,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => bail!("探索スレッドが異常終了しました"),
                }
            }
            self.advance();
            self.draw()?;

            // 読み込み中と探索中は、経過を表示し続けるために短い間隔で描き直す
            let animating = matches!(self.screen, Screen::Loading) || self.pending.is_some();
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
    let (requests, replies, engine) = spawn_engine(Arc::clone(&setup), Arc::clone(&stop));
    let mut app = App::new(setup, requests);
    let terminal = Terminal::enter()?;
    let result = app.event_loop(&replies);
    drop(terminal);

    // 推論の最中にプロセスを終えると onnxruntime の後始末と競合して落ちるので、探索スレッドの終了を待つ
    stop.store(true, Ordering::Relaxed);
    let App { requests, game, mut records, .. } = app;
    drop(requests);
    let _ = engine.join();

    if !game.moves.is_empty() {
        records.push(game.record());
    }
    for record in &records {
        println!("棋譜: {record}");
    }
    result
}
