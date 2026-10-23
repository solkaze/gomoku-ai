//! 自己対局データ (*.npz) の特徴平面を、今の features.rs の定義で作り直す。特徴平面を増やしたとき
//! (例: VCF の平面を加えたとき) に、それまでのデータを学習に使い続けるためのもの。
//!
//!     cargo run --release --example refeature -- runs/selfplay [--threads N]
//!
//! 局面は平面 0〜2 (手番側の石・相手の石・手番) から復元し、方策と価値はそのまま残す。既に今の平面数のファイルは
//! 飛ばす。学習ループは更新時刻で新しいデータを判断するので、書き直したファイルの更新時刻は元に戻す。
//! 学習ループを止めてから実行すること。

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use anyhow::{Context, bail};
use gomoku_engine::features::{FEATURE_LEN, PLANES, encode};
use gomoku_engine::npz::{NpyArray, NpyData, write_npz};
use gomoku_engine::{Board, CELLS, SIZE, Stone};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(args.first().context("自己対局データのディレクトリを指定してください")?);
    let threads = match args.iter().position(|a| a == "--threads") {
        Some(i) => args.get(i + 1).context("--threads の値がありません")?.parse()?,
        None => thread::available_parallelism().map_or(4, |n| n.get()),
    };
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.retain(|p| p.extension().is_some_and(|e| e == "npz"));
    paths.sort();

    let next = AtomicUsize::new(0);
    let (converted, positions, skipped) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
    let errors = Mutex::new(Vec::new());
    thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(path) = paths.get(i) else { break };
                    match convert(path) {
                        Ok(Some(n)) => {
                            positions.fetch_add(n, Ordering::Relaxed);
                            let done = converted.fetch_add(1, Ordering::Relaxed) + 1;
                            if done % 500 == 0 {
                                println!("{done} ファイル書き直しました ({} / {})", i + 1, paths.len());
                            }
                        }
                        Ok(None) => {
                            skipped.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => errors.lock().unwrap().push(format!("{}: {e:#}", path.display())),
                    }
                }
            });
        }
    });
    println!(
        "{} ファイル ({} 局面) を {PLANES} 平面に書き直しました。既に {PLANES} 平面だったもの {} ファイル",
        converted.into_inner(),
        positions.into_inner(),
        skipped.into_inner()
    );
    let errors = errors.into_inner().unwrap();
    for e in &errors {
        eprintln!("{e}");
    }
    if !errors.is_empty() {
        bail!("{} ファイルを書き直せませんでした", errors.len());
    }
    Ok(())
}

/// path を書き直し、局面数を返す。既に今の平面数なら None。
fn convert(path: &Path) -> anyhow::Result<Option<usize>> {
    let mtime = std::fs::metadata(path)?.modified()?;
    let states = read_npy(path, "states")?;
    let policy = read_npy(path, "policy")?;
    let value = read_npy(path, "value")?;
    let n = value.len() / 4;
    if n == 0 || states.len() % (n * CELLS) != 0 || policy.len() != n * CELLS * 4 {
        bail!("配列の大きさが合いません");
    }
    let planes = states.len() / (n * CELLS);
    if planes == PLANES {
        return Ok(None);
    }
    if planes < 3 {
        bail!("平面数 {planes} からは局面を復元できません");
    }
    let mut features = vec![0; n * FEATURE_LEN];
    for (old, new) in states.chunks_exact(planes * CELLS).zip(features.chunks_exact_mut(FEATURE_LEN)) {
        // 平面 0: 手番側の石 / 1: 相手の石 / 2: 手番が黒なら 1
        let side = if old[2 * CELLS] == 1 { Stone::Black } else { Stone::White };
        let own: Vec<usize> = (0..CELLS).filter(|&p| old[p] == 1).collect();
        let opp: Vec<usize> = (0..CELLS).filter(|&p| old[CELLS + p] == 1).collect();
        let (black, white) = if side == Stone::Black { (own, opp) } else { (opp, own) };
        encode(&Board::from_stones(&black, &white, side), new);
    }
    let to_f32 = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() };
    let (policy, value) = (to_f32(&policy), to_f32(&value));
    write_npz(
        path,
        &[
            NpyArray { name: "states", shape: vec![n, PLANES, SIZE, SIZE], data: NpyData::U8(&features) },
            NpyArray { name: "policy", shape: vec![n, CELLS], data: NpyData::F32(&policy) },
            NpyArray { name: "value", shape: vec![n], data: NpyData::F32(&value) },
        ],
    )?;
    File::options().write(true).open(path)?.set_modified(mtime)?;
    Ok(Some(n))
}

/// npz の中の name.npy のデータ部分 (ヘッダを除いたバイト列)。
fn read_npy(path: &Path, name: &str) -> anyhow::Result<Vec<u8>> {
    let mut zip = zip::ZipArchive::new(File::open(path)?)?;
    let mut bytes = Vec::new();
    zip.by_name(&format!("{name}.npy"))?.read_to_end(&mut bytes)?;
    if !bytes.starts_with(b"\x93NUMPY") {
        bail!("{name} が npy ではありません");
    }
    // バージョン 1 はヘッダ長が 2 バイト、2 以降は 4 バイト
    let start = match bytes[6] {
        1 => 10 + u16::from_le_bytes([bytes[8], bytes[9]]) as usize,
        _ => 12 + u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
    };
    Ok(bytes.split_off(start))
}
