//! 行単位の表示での対局。座標を「x y」で入力する。

use std::io::{self, BufRead, Write};
use std::time::Instant;

use gomoku_engine::inference::LoadStage;
use gomoku_engine::{Board, CELLS, CENTER, GameResult, Rules, SIZE, SearchResult, Stone, pos};

use crate::{Engine, Game, Mode, RULES_WARNING, Setup, clock, coord, forbidden_note, record, top_candidates};

const HELP: &str = "\
  7 7 など    その点に打つ (x y の順。x は左から、y は上から、どちらも 0〜14)
  undo        自分の直前の手まで戻す (自由対局では 1 手戻す)
  hint        今の局面での AI の読みを表示する (自由対局では使えない)
  help        この説明
  quit        終了";

/// 星 (盤上の目印)
const STARS: [usize; 5] = [pos(3, 3), pos(3, 11), CENTER, pos(11, 3), pos(11, 11)];

fn parse_coord(s: &str) -> Option<usize> {
    let mut nums = s.split_whitespace().map(|t| t.parse::<usize>().ok().filter(|&n| n < SIZE));
    let (x, y) = (nums.next()??, nums.next()??);
    if nums.next().is_some() {
        return None;
    }
    Some(pos(y, x))
}

/// 人が先手 (黒) と後手 (白) のどちらを持つか、または自由対局にするかを尋ねる。入力が尽きたら None。
fn ask_mode(stdin: &mut impl BufRead) -> io::Result<Option<Mode>> {
    loop {
        print!("先手 (黒) と後手 (白) のどちらを持ちますか? [1: 先手 / 2: 後手 / 3: 自由対局 (AI なし)] > ");
        io::stdout().flush()?;
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            return Ok(None);
        }
        match line.trim() {
            "1" | "先手" | "黒" => return Ok(Some(Mode::VsAi(Stone::Black))),
            "2" | "後手" | "白" => return Ok(Some(Mode::VsAi(Stone::White))),
            "3" | "自由" | "自由対局" => return Ok(Some(Mode::Free)),
            _ => println!("1〜3 を入力してください"),
        }
    }
}

fn stone_name(stone: Stone) -> &'static str {
    match stone {
        Stone::Black => "黒 (X)",
        Stone::White => "白 (O)",
    }
}

/// 盤を表示する。直前の手は [ ] で囲み、forbidden が与えられたら黒の禁じ手 (打つと負け) を ! で示す。
fn render(board: &Board, forbidden: Option<&[bool; CELLS]>) -> String {
    let mut out = String::from("y\\x");
    for c in 0..SIZE {
        out.push_str(&format!("{c:^3}"));
    }
    out.push('\n');
    for r in 0..SIZE {
        out.push_str(&format!("{r:>2} "));
        for c in 0..SIZE {
            let p = pos(r, c);
            let mark = match board.get(p) {
                Some(Stone::Black) => 'X',
                Some(Stone::White) => 'O',
                None if forbidden.is_some_and(|f| f[p]) => '!',
                None if STARS.contains(&p) => '+',
                None => '.',
            };
            if board.last_move() == Some(p) {
                out.push_str(&format!("[{mark}]"));
            } else {
                out.push_str(&format!(" {mark} "));
            }
        }
        out.push_str(&format!(" {r}\n"));
    }
    out
}

/// 探索結果の評価値と、訪問回数の多い候補手を表示する。
fn print_analysis(result: &SearchResult, candidates: usize) {
    println!("  評価値 {:+.2} (手番側の勝率換算 {:.0}%)", result.value, (result.value + 1.0) * 50.0);
    if candidates == 0 {
        return;
    }
    let line: Vec<String> =
        top_candidates(result, candidates).iter().map(|&(mv, pct)| format!("({}) {pct:.0}%", coord(mv))).collect();
    println!("  候補: {}", line.join(", "));
}

pub fn run(setup: Setup) -> anyhow::Result<()> {
    // 自由対局だけなら AI は要らないので、モデルを読み込まない
    let mut engine = if setup.mode == Some(Mode::Free) {
        None
    } else {
        println!("モデル: {} ({:?}, 探索 {} 回/手)", setup.model_path.display(), setup.inference.device, setup.mcts.simulations);
        let stages = LoadStage::stages(setup.inference.device);
        let start = Instant::now();
        let engine = Engine::load(&setup, &mut |stage| {
            let n = stages.iter().position(|&s| s == stage).map_or(0, |i| i + 1);
            println!("[{n}/{}] {}...", stages.len(), stage.label());
        })?;
        println!("読み込み完了 ({:.1} 秒)", start.elapsed().as_secs_f64());
        Some(engine)
    };

    let mut stdin = io::stdin().lock();
    let mode = match setup.mode {
        Some(mode) => mode,
        None => match ask_mode(&mut stdin)? {
            Some(mode) => mode,
            None => return Ok(()),
        },
    };
    let intro = match mode {
        Mode::VsAi(human) => format!("あなたは{}です。", stone_name(human)),
        Mode::Free => "自由対局です (AI なし)。黒と白を交互に打ちます。".to_string(),
    };
    let rules = setup.rules;
    let mark = if rules == Rules::none() { "" } else { " (! で表示)" };
    println!("{intro}黒: {}{mark}。", forbidden_note(rules));
    if rules != Rules::default() {
        if !rules.overline {
            println!("長連が禁じ手でないので、黒も六以上の連で勝ちになります。");
        }
        if matches!(mode, Mode::VsAi(_)) {
            println!("注意: {RULES_WARNING}。");
        }
    }
    println!("{HELP}\n");

    let mut game = Game::new(rules);
    'game: loop {
        let side = game.board.side_to_move();
        let forbidden = (side == Stone::Black && mode.is_human(side)).then(|| game.board.forbidden_mask());
        println!("\n{}", render(&game.board, forbidden.as_ref()));

        match game.board.result() {
            GameResult::Ongoing => {}
            GameResult::Win(winner) => {
                let note = if game.board.lost_by_forbidden() { " (黒の禁じ手)" } else { "" };
                let name = match mode {
                    Mode::VsAi(human) if winner == human => "あなた",
                    Mode::VsAi(_) => "AI ",
                    Mode::Free => stone_name(winner),
                };
                println!("{name}の勝ちです{note}");
                break 'game;
            }
            GameResult::Draw => {
                println!("引き分けです");
                break 'game;
            }
        }

        if !mode.is_human(side) {
            let engine = engine.as_mut().expect("AI との対戦ではモデルを読み込んでいる");
            let start = Instant::now();
            let result = engine.search(&game.board);
            let mv = result.best_move();
            println!("AI ({}): ({}) ({:.1} 秒)", stone_name(side), coord(mv), start.elapsed().as_secs_f64());
            print_analysis(&result, setup.candidates);
            game.play_ai(mv, &result).expect("探索結果は合法手");
            continue;
        }

        if game.board.move_count() == 0 {
            println!("黒の初手は天元 (7 7) に固定です");
            game.play(CENTER).expect("天元は合法手");
            continue;
        }

        loop {
            print!("{} {}手目 [対局 {}] > ", stone_name(side), game.board.move_count() + 1, clock(game.elapsed()));
            io::stdout().flush()?;
            let mut line = String::new();
            if stdin.read_line(&mut line)? == 0 {
                break 'game;
            }
            match line.trim() {
                "" => continue,
                "quit" | "exit" => break 'game,
                "help" | "?" => println!("{HELP}"),
                "undo" => {
                    if !game.undo(mode) {
                        println!("戻せる手がありません");
                        continue;
                    }
                    break;
                }
                "hint" => match (mode, engine.as_mut()) {
                    (Mode::VsAi(_), Some(engine)) => {
                        let result = engine.search(&game.board);
                        print_analysis(&result, setup.candidates.max(5));
                    }
                    _ => println!("自由対局ではヒントは使えません"),
                },
                input => match parse_coord(input) {
                    None => println!("座標は 7 7 のように 0〜14 の数字 2 つを半角スペース区切りで入力してください (help で説明)"),
                    Some(p) => match game.play(p) {
                        Ok(_) => {
                            let who = match mode {
                                Mode::VsAi(_) => format!("あなた ({})", stone_name(side)),
                                Mode::Free => stone_name(side).to_string(),
                            };
                            let secs = game.times.last().map_or(0.0, |t| t.as_secs_f64());
                            println!("{who}: ({}) ({secs:.1} 秒)", coord(p));
                            break;
                        }
                        Err(e) => println!("({}): {e}", coord(p)),
                    },
                },
            }
        }
    }

    if !game.moves.is_empty() {
        println!("\n{}", game.record());
    }
    match record::save(&setup, mode, &mut game) {
        Ok(Some(path)) => println!("記録を書き出しました: {}", path.display()),
        Ok(None) => {}
        Err(e) => println!("記録を書き出せませんでした: {e:#}"),
    }
    Ok(())
}
