//! 対局の記録 (棋譜・各手の時間・AI の読み) を 1 局 1 ファイルの JSON で書き出す。
//!
//! ファイル名は対局を始めた時刻 (例: 20261016-140312.json)。同じ対局は同じファイルに上書きするので、
//! 終局後に待ったをして続けた場合も、最後に書き出した時点の内容が残る。

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use gomoku_engine::{GameResult, SIZE, Stone};
use serde::Serialize;

use crate::{Forbidden, Game, Mode, Setup};

#[derive(Serialize)]
struct Record {
    /// 対局を始めた時刻 (ローカル時刻)
    started: String,
    /// AI のモデル (自由対局では null)
    model: Option<String>,
    simulations: Option<u32>,
    /// 負けの局面で終局を遠ざける強さ (MctsConfig::moves_left_effect)
    moves_left_effect: Option<f32>,
    /// 黒に適用した禁じ手
    forbidden: Vec<&'static str>,
    /// 人が持った石 ("black" / "white"。自由対局では "both")
    human: &'static str,
    /// "black" / "white" (勝った側) / "draw" / "unfinished" (途中で終えた)
    result: &'static str,
    /// 黒の禁じ手で決着したか
    lost_by_forbidden: bool,
    /// 対局時間と、黒・白それぞれが使った時間 (秒)
    seconds: Seconds,
    /// 画面に表示するのと同じ形式の棋譜
    kifu: String,
    moves: Vec<MoveRecord>,
}

#[derive(Serialize)]
struct Seconds {
    total: f64,
    black: f64,
    white: f64,
}

#[derive(Serialize)]
struct MoveRecord {
    /// 何手目か (1 始まり)
    n: usize,
    stone: &'static str,
    /// "human" / "ai" (黒の初手は天元に固定なので "fixed")
    by: &'static str,
    x: usize,
    y: usize,
    secs: f64,
    /// AI が打った手の読み
    #[serde(skip_serializing_if = "Option::is_none")]
    ai: Option<AiRecord>,
}

#[derive(Serialize)]
struct AiRecord {
    /// 手番側 (AI) から見た評価値 [-1, 1] と、勝率に換算した値 (%)
    value: f64,
    win_rate: f64,
    candidates: Vec<Candidate>,
}

#[derive(Serialize)]
struct Candidate {
    x: usize,
    y: usize,
    /// 訪問回数の割合 (%)
    share: f64,
}

fn stone_key(stone: Stone) -> &'static str {
    match stone {
        Stone::Black => "black",
        Stone::White => "white",
    }
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn build(setup: &Setup, mode: Mode, game: &Game) -> Record {
    let ai = mode != Mode::Free;
    let model = setup.model_path.file_name().unwrap_or(setup.model_path.as_os_str()).to_string_lossy().into_owned();
    let result = match game.board.result() {
        GameResult::Win(winner) => stone_key(winner),
        GameResult::Draw => "draw",
        GameResult::Ongoing => "unfinished",
    };
    let moves = game
        .moves
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            // パスはないので、黒の手は偶数番目 (0 始まり)、白の手は奇数番目
            let stone = if i % 2 == 0 { Stone::Black } else { Stone::White };
            let note = game.notes[i].as_ref();
            let by = match note {
                Some(_) => "ai",
                None if i == 0 && !mode.is_human(Stone::Black) => "ai",
                None if i == 0 => "fixed",
                None => "human",
            };
            MoveRecord {
                n: i + 1,
                stone: stone_key(stone),
                by,
                x: p % SIZE,
                y: p / SIZE,
                secs: round1(game.times[i].as_secs_f64()),
                ai: note.map(|note| AiRecord {
                    value: (note.value as f64 * 1000.0).round() / 1000.0,
                    win_rate: round1((note.value as f64 + 1.0) * 50.0),
                    candidates: note
                        .candidates
                        .iter()
                        .map(|&(mv, share)| Candidate { x: mv % SIZE, y: mv / SIZE, share: round1(share) })
                        .collect(),
                }),
            }
        })
        .collect();
    Record {
        started: local_time(game.started_at, "%Y-%m-%d %H:%M:%S"),
        model: ai.then_some(model),
        simulations: ai.then_some(setup.mcts.simulations),
        moves_left_effect: ai.then_some(setup.mcts.moves_left_effect),
        forbidden: Forbidden::ALL.into_iter().filter(|f| f.enabled(game.board.rules())).map(Forbidden::name).collect(),
        human: match mode {
            Mode::VsAi(human) => stone_key(human),
            Mode::Free => "both",
        },
        result,
        lost_by_forbidden: game.board.lost_by_forbidden(),
        seconds: Seconds {
            total: round1(game.elapsed().as_secs_f64()),
            black: round1(game.total(Stone::Black).as_secs_f64()),
            white: round1(game.total(Stone::White).as_secs_f64()),
        },
        kifu: game.record(),
        moves,
    }
}

/// 対局の記録を setup.record_dir に書き出し、書き出したファイルを返す。
/// 書き出さない設定のときと、まだ 1 手も打っていないときは何もしない。
pub fn save(setup: &Setup, mode: Mode, game: &mut Game) -> anyhow::Result<Option<PathBuf>> {
    let Some(dir) = &setup.record_dir else {
        return Ok(None);
    };
    if game.moves.is_empty() {
        return Ok(None);
    }
    let path = match &game.record_file {
        Some(path) => path.clone(),
        None => {
            std::fs::create_dir_all(dir)?;
            let path = unused_path(dir, &local_time(game.started_at, "%Y%m%d-%H%M%S"));
            game.record_file = Some(path.clone());
            path
        }
    };
    let json = serde_json::to_string_pretty(&build(setup, mode, game))?;
    // 書きかけのファイルが残らないよう、別名で書いてから置き換える
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json + "\n")?;
    std::fs::rename(&tmp, &path)?;
    Ok(Some(path))
}

/// dir/stem.json がなければそれを、あれば dir/stem-2.json, -3, ... の空いているものを返す。
fn unused_path(dir: &Path, stem: &str) -> PathBuf {
    let mut path = dir.join(format!("{stem}.json"));
    let mut n = 2;
    while path.exists() {
        path = dir.join(format!("{stem}-{n}.json"));
        n += 1;
    }
    path
}

/// 時刻をローカル時刻で strftime の形式に整える。
fn local_time(t: SystemTime, format: &str) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()) as libc::time_t;
    let format = std::ffi::CString::new(format).expect("形式に NUL はない");
    let mut buf = [0u8; 64];
    // SAFETY: tm は localtime_r が埋める。buf の長さを strftime に渡しているので、はみ出して書かれない。
    let len = unsafe {
        let mut tm: libc::tm = std::mem::zeroed();
        if libc::localtime_r(&secs, &mut tm).is_null() {
            return secs.to_string();
        }
        libc::strftime(buf.as_mut_ptr().cast(), buf.len(), format.as_ptr(), &tm)
    };
    String::from_utf8_lossy(&buf[..len]).into_owned()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{Duration, UNIX_EPOCH};

    use gomoku_engine::config::{Device, InferenceConfig};
    use gomoku_engine::{CENTER, MctsConfig, Rules, SearchResult, Stone, pos};

    use super::*;

    fn setup(dir: &Path) -> Setup {
        Setup {
            model_path: PathBuf::from("/models/best.onnx"),
            record_dir: Some(dir.to_path_buf()),
            inference: InferenceConfig {
                device: Device::Cpu,
                max_batch: 1,
                threads: 1,
                onnxruntime_dir: PathBuf::new(),
                cuda_lib_dirs: Vec::new(),
            },
            mcts: MctsConfig { simulations: 1600, ..Default::default() },
            mode: None,
            rules: Rules::default(),
            candidates: 5,
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("gomoku-record-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn saves_moves_times_and_ai_analysis_to_one_file_per_game() {
        let dir = temp_dir("save");
        let setup = setup(&dir);
        let mode = Mode::VsAi(Stone::Black);
        let mut game = Game::new(Rules::default());
        game.started_at = UNIX_EPOCH + Duration::from_secs(1_800_000_000);
        game.play(CENTER).unwrap();
        let search = SearchResult { visits: vec![(pos(6, 6), 30), (pos(6, 7), 10)], value: 0.5 };
        game.play_ai(pos(6, 6), &search).unwrap();
        game.play(pos(7, 8)).unwrap();

        let path = save(&setup, mode, &mut game).unwrap().unwrap();
        assert_eq!(path.parent(), Some(dir.as_path()));
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(json["model"], "best.onnx");
        assert_eq!(json["simulations"], 1600);
        assert_eq!(json["human"], "black");
        assert_eq!(json["result"], "unfinished");
        assert_eq!(json["forbidden"].as_array().unwrap().len(), 3);
        let moves = json["moves"].as_array().unwrap();
        assert_eq!(moves.len(), 3);
        assert_eq!((moves[0]["by"].as_str(), moves[0]["x"].as_u64(), moves[0]["y"].as_u64()), (Some("fixed"), Some(7), Some(7)));
        assert_eq!(moves[1]["by"], "ai");
        assert_eq!(moves[1]["stone"], "white");
        assert_eq!(moves[1]["ai"]["win_rate"], 75.0);
        assert_eq!(moves[1]["ai"]["candidates"][0]["share"], 75.0);
        assert_eq!((moves[2]["by"].as_str(), moves[2]["x"].as_u64(), moves[2]["y"].as_u64()), (Some("human"), Some(8), Some(7)));
        assert!(moves[2].get("ai").is_none());

        // 待ったをして続けても同じファイルに上書きし、戻した手の読みは残さない
        assert!(game.undo(mode));
        game.play(pos(8, 8)).unwrap();
        assert_eq!(save(&setup, mode, &mut game).unwrap(), Some(path.clone()));
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(json["moves"][2]["y"], 8);
        assert_eq!(game.notes.len(), game.moves.len());

        // 同じ時刻に始めた別の対局は別のファイルにする
        let mut other = Game::new(Rules::default());
        other.started_at = game.started_at;
        other.play(CENTER).unwrap();
        let other_path = save(&setup, mode, &mut other).unwrap().unwrap();
        assert_ne!(other_path, path);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn nothing_is_saved_without_moves_or_when_disabled() {
        let dir = temp_dir("skip");
        let mut setup = setup(&dir);
        let mut game = Game::new(Rules::default());
        assert_eq!(save(&setup, Mode::Free, &mut game).unwrap(), None);
        game.play(CENTER).unwrap();
        setup.record_dir = None;
        assert_eq!(save(&setup, Mode::Free, &mut game).unwrap(), None);
        assert!(!dir.exists());
    }
}
