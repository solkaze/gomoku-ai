//! VCF (四を打ち続けて勝つ手順) の探索。
//!
//! ネットワークの方策が低く見積もった強制手順は、MCTS では読み落としやすい。探索のルートでこれを使い、
//! 自分に VCF があれば確実に打ち、相手に VCF を許す手は候補から外す (mcts.rs)。
//!
//! 攻める側は毎手で四を作り、受ける側はその四を止めるしかないものとして読む。受ける側が止めた手で四を作ったら、
//! 攻める側はそれを止める手が四になるときだけ続けられる。黒は禁じ手を打てず、禁じ手の点では四を止められない。

use std::collections::HashSet;

use crate::board::{Board, GameResult, Stone};

/// 1 回の探索で調べる局面数の上限。超えたら VCF はないものとして扱う。
pub const DEFAULT_BUDGET: u32 = 20_000;

/// 手番側の VCF を探す。見つかれば手順 (手番側の手と相手の受けを交互に並べ、最後は五の手) を返す。
pub fn find(board: &Board, budget: u32) -> Option<Vec<usize>> {
    if board.result() != GameResult::Ongoing {
        return None;
    }
    let me = board.side_to_move();
    if let Some(&win) = board.winning_moves(me).first() {
        return Some(vec![win]);
    }
    let threats = board.winning_moves(me.opponent());
    let mut nodes = 0;
    search(board, &threats, &mut nodes, budget, &mut HashSet::new())
}

/// moves のうち、打った後に相手に VCF を許さない手。全ての手が許すなら空。
/// 読み切れなかった (局面数の上限に達した) 手は許さないものとして残す。
pub fn safe_moves(board: &Board, moves: &[usize], budget: u32) -> Vec<usize> {
    moves
        .iter()
        .copied()
        .filter(|&p| {
            let mut after = board.clone();
            match after.play(p) {
                Ok(GameResult::Ongoing) => find(&after, budget).is_none(),
                // 勝ちは安全、禁じ手での負けや打てない手は安全でない
                Ok(GameResult::Win(winner)) => winner == board.side_to_move(),
                Ok(GameResult::Draw) => true,
                Err(_) => false,
            }
        })
        .collect()
}

/// line (find が返す形の手順) が board でもそのまま VCF として成り立つか。find と同じ規則で確かめる。
/// 探し直すより速いので、局面が少し変わったときに前の手順が使えるかを調べるのに使う (vct.rs)。
pub fn verify(board: &Board, line: &[usize]) -> bool {
    if board.result() != GameResult::Ongoing {
        return false;
    }
    let me = board.side_to_move();
    let opponent = me.opponent();
    let mut b = board.clone();
    // 相手の五の点。止めた点は打った時点で空点でなくなるので、確かめる前に除く
    let mut threats = b.winning_moves(opponent);
    for step in line.chunks(2) {
        let p = step[0];
        threats.retain(|&q| b.get(q).is_none());
        if threats.iter().any(|&q| q != p) {
            return false;
        }
        if me == Stone::Black && b.is_forbidden(p) {
            return false;
        }
        match b.play(p) {
            Ok(GameResult::Win(winner)) => return winner == me,
            Ok(GameResult::Ongoing) => {}
            _ => return false,
        }
        let block = match b.winning_moves_through(me, p).as_slice() {
            [] => return false,
            &[block] => block,
            _ => return true,
        };
        if opponent == Stone::Black && b.is_forbidden(block) {
            return true;
        }
        if step.get(1) != Some(&block) || !matches!(b.play(block), Ok(GameResult::Ongoing)) {
            return false;
        }
        threats = b.winning_moves_through(opponent, block);
    }
    false
}

/// 手番側に五の点がない局面から VCF を探す。threats は相手の五の点。
/// 盤全体から五の点を探し直すのは遅いので、四とその止めで変わる分だけを調べる: 手番側の四は毎回止められるので
/// 手番側の五の点は残らず、相手の五の点は止めた手の線上にしかできない。
/// failed: VCF がないと読み切った局面 (四を打つ順番が違うだけで同じ局面に何度も行き着くので、読み直さない)。
fn search(
    board: &Board,
    threats: &[usize],
    nodes: &mut u32,
    budget: u32,
    failed: &mut HashSet<u64>,
) -> Option<Vec<usize>> {
    let key = board.key();
    if failed.contains(&key) {
        return None;
    }
    *nodes += 1;
    if *nodes > budget {
        return None;
    }
    let me = board.side_to_move();
    let opponent = me.opponent();
    let candidates = match threats {
        [] => board.four_candidates(me),
        // 相手の四は止めるしかない。止める手が四になるときだけ続けられる
        &[block] => vec![block],
        _ => return None,
    };
    for p in candidates {
        if me == Stone::Black && board.is_forbidden(p) {
            continue;
        }
        let mut after = board.clone();
        if !matches!(after.play(p), Ok(GameResult::Ongoing)) {
            continue;
        }
        // p でできた四の五の点は、p を通る線上にある
        let fives = after.winning_moves_through(me, p);
        let block = match fives.as_slice() {
            [] => continue,
            &[block] => block,
            // 五の点が 2 つ以上 (活四・四四) は止められない
            &[first, ..] => return Some(vec![p, first]),
        };
        if opponent == Stone::Black && after.is_forbidden(block) {
            // 黒は禁じ手の点で止められない
            return Some(vec![p, block]);
        }
        if !matches!(after.play(block), Ok(GameResult::Ongoing)) {
            continue;
        }
        let threats = after.winning_moves_through(opponent, block);
        if let Some(rest) = search(&after, &threats, nodes, budget, failed) {
            let mut line = vec![p, block];
            line.extend(rest);
            return Some(line);
        }
        if *nodes > budget {
            return None;
        }
    }
    failed.insert(key);
    None
}
