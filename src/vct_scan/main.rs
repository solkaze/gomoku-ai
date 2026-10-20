//! 棋譜に VCT の読み (vct.rs) を当てて、AI が VCT を見落とした手を数える。VCT を対局の探索に組み込むかの判断材料。
//!
//!     cargo run --release --bin vct_scan -- [--games runs/games] [--selfplay-files N] [--threads N] [--out report.json]
//!
//! 調べる局面と「AI の手」:
//! - 人との対局の記録 (--games): AI が打った手。
//! - 自己対局の学習データ (--selfplay-files): 新しい順に N ファイルの全局面。打った手は記録されていないので、
//!   訪問回数が最も多い手 (方策の目標の最大の手) を AI の手とみなす。序盤の温度付きの手では実際の手と違うことがある。
//!
//! 数えるもの (手番側 = AI):
//! - 攻めの見落とし: 手番側に VCT がある (VCF はない) のに、AI の手のあとでは VCT で勝てない。
//! - 守りの見落とし: AI の手のあと、相手に VCT がある。ほかの手なら防げたもの (避けられた) と、どの手でも防げない
//!   もの (既に負け) に分ける。手番側に VCF・VCT があった局面は除く。

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

use anyhow::{Context, bail};
use clap::Parser;
use gomoku_engine::features::PLANES;
use gomoku_engine::vct::{self, Limits, Outcome};
use gomoku_engine::{Board, CELLS, GameResult, Rules, SIZE, Stone, pos, vcf};
use serde::{Deserialize, Serialize};

#[derive(Parser)]
struct Args {
    /// 人との対局の記録 (*.json) のディレクトリ
    #[arg(long, default_value = "runs/games")]
    games: PathBuf,
    /// 自己対局の学習データ (*.npz) のディレクトリ
    #[arg(long, default_value = "runs/selfplay")]
    selfplay: PathBuf,
    /// 調べる自己対局のファイル数 (新しい順。1 ファイル 100 局)
    #[arg(long, default_value_t = 1)]
    selfplay_files: usize,
    /// VCT の 1 回の読みの局面数の上限
    #[arg(long, default_value_t = Limits::default().nodes)]
    nodes: u32,
    /// VCT の攻める側の手数の上限
    #[arg(long, default_value_t = Limits::default().depth)]
    depth: u32,
    #[arg(long, default_value_t = 8)]
    threads: usize,
    /// 見落としの局面を JSON で書き出す先
    #[arg(long)]
    out: Option<PathBuf>,
}

/// 調べる 1 局面
struct Position {
    source: String,
    board: Board,
    /// AI の手
    mv: usize,
    /// 手番側 (AI) がその対局に勝ったか (分からなければ None)
    won: Option<bool>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Verdict {
    /// 手番側に VCF があった (探索のルートで VCF は既に読んでいる)
    OwnVcf,
    /// 手番側に VCT があり、AI の手も VCT の手だった
    VctPlayed,
    /// 手番側に VCT があったのに、AI の手のあとでは VCT で勝てない
    VctMissed,
    /// AI の手のあと、相手に VCT がある。ほかの手なら防げた
    AllowedAvoidable,
    /// AI の手のあと、相手に VCT がある。どの手でも防げない (既に負け)
    AllowedUnavoidable,
    /// AI の手のあと相手に VCT があるが、ほかの手で防げるかは読み切れなかった
    AllowedUnclear,
    /// どちらにも VCT はない (相手の VCT は読み切れなかったものを含む)
    Quiet,
    /// 手番側の VCT を読み切れず、相手に VCT を許してもいない
    Unknown,
}

#[derive(Serialize)]
struct Finding {
    source: String,
    verdict: Verdict,
    side: &'static str,
    /// AI の手 (x y)
    played: (usize, usize),
    /// 手番側の VCT の最初の手、または相手の VCT の最初の手 (x y)
    vct: Option<(usize, usize)>,
    /// 避けられた場合の、相手に VCT を許さない手の例 (x y)
    safe: Option<(usize, usize)>,
    won: Option<bool>,
    board: String,
}

fn xy(p: usize) -> (usize, usize) {
    (p % SIZE, p / SIZE)
}

fn stone_name(stone: Stone) -> &'static str {
    match stone {
        Stone::Black => "black",
        Stone::White => "white",
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut positions = load_games(&args.games)?;
    let human_positions = positions.len();
    positions.extend(load_selfplay(&args.selfplay, args.selfplay_files)?);
    println!(
        "局面: 人との対局 {human_positions} / 自己対局 {} (VCT の上限: {} 局面 / {} 手)",
        positions.len() - human_positions,
        args.nodes,
        args.depth
    );
    let limits = Limits { nodes: args.nodes, depth: args.depth, ..Limits::default() };

    let start = Instant::now();
    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::new());
    thread::scope(|scope| {
        for _ in 0..args.threads.max(1) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(position) = positions.get(i) else { break };
                    let t = Instant::now();
                    let (verdict, vct, safe) = examine(position, limits);
                    results.lock().unwrap().push((i, verdict, vct, safe, t.elapsed().as_secs_f64()));
                    if (i + 1) % 200 == 0 {
                        println!("  {} / {} 局面 ({:.0} 秒)", i + 1, positions.len(), start.elapsed().as_secs_f64());
                    }
                }
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|r| r.0);

    for (label, range) in [("人との対局", 0..human_positions), ("自己対局", human_positions..positions.len())] {
        let rows: Vec<_> = results.iter().filter(|r| range.contains(&r.0)).collect();
        if rows.is_empty() {
            continue;
        }
        println!("\n## {label} ({} 局面)", rows.len());
        for verdict in [
            Verdict::OwnVcf,
            Verdict::VctPlayed,
            Verdict::VctMissed,
            Verdict::AllowedAvoidable,
            Verdict::AllowedUnavoidable,
            Verdict::AllowedUnclear,
            Verdict::Unknown,
            Verdict::Quiet,
        ] {
            let hits: Vec<_> = rows.iter().filter(|r| r.1 == verdict).collect();
            let won = hits.iter().filter(|r| positions[r.0].won == Some(true)).count();
            println!("{:<20} {:>6} 局面 (その対局に勝った {won})", format!("{verdict:?}"), hits.len());
        }
        let mut secs: Vec<f64> = rows.iter().map(|r| r.4).collect();
        secs.sort_by(f64::total_cmp);
        println!(
            "1 局面の時間: 中央値 {:.2} 秒 / 90% {:.2} 秒 / 最大 {:.2} 秒",
            secs[secs.len() / 2],
            secs[secs.len() * 9 / 10],
            secs[secs.len() - 1]
        );
    }
    println!("\n合計 {:.0} 秒", start.elapsed().as_secs_f64());

    if let Some(out) = &args.out {
        let findings: Vec<Finding> = results
            .iter()
            .filter(|r| !matches!(r.1, Verdict::Quiet | Verdict::OwnVcf | Verdict::VctPlayed))
            .map(|&(i, verdict, vct, safe, _)| {
                let p = &positions[i];
                Finding {
                    source: p.source.clone(),
                    verdict,
                    side: stone_name(p.board.side_to_move()),
                    played: xy(p.mv),
                    vct: vct.map(xy),
                    safe: safe.map(xy),
                    won: p.won,
                    board: p.board.to_string(),
                }
            })
            .collect();
        std::fs::write(out, serde_json::to_string_pretty(&findings)?)?;
        println!("書き出しました: {}", out.display());
    }
    Ok(())
}

/// (判定, VCT の最初の手, 相手の VCT を防げる手の例)
fn examine(position: &Position, limits: Limits) -> (Verdict, Option<usize>, Option<usize>) {
    let board = &position.board;
    if vcf::find(board, vcf::DEFAULT_BUDGET).is_some() {
        return (Verdict::OwnVcf, None, None);
    }
    let own = vct::solve(board, limits);
    if let Outcome::Win(first) = own {
        let verdict = match vct::wins_with(board, position.mv, limits) {
            Outcome::Win(_) => Verdict::VctPlayed,
            _ => Verdict::VctMissed,
        };
        return (verdict, Some(first), None);
    }
    // 手番側の VCT を読み切れなくても、相手に VCT を許したかは調べる
    let quiet = if own == Outcome::Unknown { Verdict::Unknown } else { Verdict::Quiet };
    let Some(after) = play(board, position.mv) else { return (quiet, None, None) };
    let Outcome::Win(threat) = vct::solve(&after, limits) else { return (quiet, None, None) };
    // ほかの手で相手の VCT を防げるか
    let mut unclear = false;
    for alt in nearby_moves(board) {
        if alt == position.mv {
            continue;
        }
        let Some(after) = play(board, alt) else { continue };
        match vct::solve(&after, limits) {
            Outcome::NoWin => return (Verdict::AllowedAvoidable, Some(threat), Some(alt)),
            Outcome::Unknown => unclear = true,
            Outcome::Win(_) => {}
        }
    }
    let verdict = if unclear { Verdict::AllowedUnclear } else { Verdict::AllowedUnavoidable };
    (verdict, Some(threat), None)
}

/// 手番側が p に打った局面。決着がつく手・打てない手・黒の禁じ手なら None。
fn play(board: &Board, p: usize) -> Option<Board> {
    if board.side_to_move() == Stone::Black && board.is_forbidden(p) {
        return None;
    }
    let mut after = board.clone();
    matches!(after.play(p), Ok(GameResult::Ongoing)).then_some(after)
}

/// 既にある石から 2 路以内の空点 (防ぐ手の候補)。
fn nearby_moves(board: &Board) -> Vec<usize> {
    let near = |p: usize| {
        let (r, c) = ((p / SIZE) as isize, (p % SIZE) as isize);
        (-2..=2).any(|dr| {
            (-2..=2).any(|dc| {
                let (rr, cc) = (r + dr, c + dc);
                (0..SIZE as isize).contains(&rr)
                    && (0..SIZE as isize).contains(&cc)
                    && board.get(pos(rr as usize, cc as usize)).is_some()
            })
        })
    };
    (0..CELLS).filter(|&p| board.get(p).is_none() && near(p)).collect()
}

#[derive(Deserialize)]
struct Record {
    forbidden: Vec<String>,
    result: String,
    moves: Vec<RecordMove>,
}

#[derive(Deserialize)]
struct RecordMove {
    n: usize,
    stone: String,
    by: String,
    x: usize,
    y: usize,
}

fn load_games(dir: &Path) -> anyhow::Result<Vec<Position>> {
    let mut paths: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(entries) => entries.filter_map(|e| e.ok().map(|e| e.path())).collect(),
        Err(_) => return Ok(Vec::new()),
    };
    paths.retain(|p| p.extension().is_some_and(|e| e == "json"));
    paths.sort();
    let mut positions = Vec::new();
    for path in paths {
        let record: Record = serde_json::from_reader(File::open(&path)?)
            .with_context(|| format!("{} を読めません", path.display()))?;
        let has = |name: &str| record.forbidden.iter().any(|f| f == name);
        let rules = Rules { double_three: has("三三"), double_four: has("四四"), overline: has("長連") };
        let mut board = Board::with_rules(rules);
        let name = path.file_name().unwrap().to_string_lossy();
        for m in &record.moves {
            let p = pos(m.y, m.x);
            if m.by == "ai" && board.move_count() > 0 {
                let won = match record.result.as_str() {
                    "black" | "white" => Some(record.result == m.stone),
                    _ => None,
                };
                positions.push(Position { source: format!("{name} {} 手目", m.n), board: board.clone(), mv: p, won });
            }
            if board.play(p).is_err() {
                break;
            }
        }
    }
    Ok(positions)
}

fn load_selfplay(dir: &Path, files: usize) -> anyhow::Result<Vec<Position>> {
    if files == 0 {
        return Ok(Vec::new());
    }
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.retain(|p| p.extension().is_some_and(|e| e == "npz"));
    // ファイル名は「書き出し開始の時刻_連番」なので、名前順が書き出し順
    paths.sort();
    let mut positions = Vec::new();
    for path in paths.iter().rev().take(files) {
        let name = path.file_name().unwrap().to_string_lossy();
        let states = read_npy(path, "states")?;
        let policy: Vec<f32> =
            read_npy(path, "policy")?.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let value: Vec<f32> =
            read_npy(path, "value")?.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        if states.len() != value.len() * PLANES * CELLS || policy.len() != value.len() * CELLS {
            bail!("{} の配列の大きさが合いません", path.display());
        }
        for (i, features) in states.chunks_exact(PLANES * CELLS).enumerate() {
            // 特徴平面 (features.rs): 0 手番側の石 / 1 相手の石 / 2 手番が黒なら 1
            let side = if features[2 * CELLS] == 1 { Stone::Black } else { Stone::White };
            let own: Vec<usize> = (0..CELLS).filter(|&p| features[p] == 1).collect();
            let opp: Vec<usize> = (0..CELLS).filter(|&p| features[CELLS + p] == 1).collect();
            let (black, white) = if side == Stone::Black { (own, opp) } else { (opp, own) };
            let board = Board::from_stones(&black, &white, side);
            let pi = &policy[i * CELLS..(i + 1) * CELLS];
            let mv = (0..CELLS).max_by(|&a, &b| pi[a].total_cmp(&pi[b])).unwrap();
            let won = match value[i] {
                v if v > 0.5 => Some(true),
                v if v < -0.5 => Some(false),
                _ => None,
            };
            positions.push(Position { source: format!("{name} #{i}"), board, mv, won });
        }
    }
    Ok(positions)
}

/// npz の中の name.npy のデータ部分 (ヘッダを除いたバイト列)。
fn read_npy(path: &Path, name: &str) -> anyhow::Result<Vec<u8>> {
    let mut zip = zip::ZipArchive::new(File::open(path)?)?;
    let mut bytes = Vec::new();
    zip.by_name(&format!("{name}.npy"))?.read_to_end(&mut bytes)?;
    if !bytes.starts_with(b"\x93NUMPY") {
        bail!("{} の {name} が npy ではありません", path.display());
    }
    // バージョン 1 はヘッダ長が 2 バイト、2 以降は 4 バイト
    let start = match bytes[6] {
        1 => 10 + u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
        _ => 12 + u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
    };
    Ok(bytes.split_off(start))
}
