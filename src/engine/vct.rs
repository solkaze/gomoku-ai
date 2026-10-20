//! VCT (四と三で攻め続けて勝つ手順) の探索。
//!
//! VCF (vcf.rs) は四だけで攻める手順しか読まない。三を使った攻めを読まないと、相手の勝ち筋を崩す受けを選べない。
//!
//! 攻める側は毎手で四か三を作り、受ける側はそれに応じるしかないものとして読む。
//! - 四には五の点を止めるしかない。五の点が 2 つ以上あるか、黒が禁じ手の点でしか止められなければ攻める側の勝ち。
//! - 三は「受ける側が一手休むと攻める側に VCF がある」状態として扱う。受ける側の手は、受ける側の四と、
//!   打った後に攻める側の VCF がなくなる点。後者は、見つけた VCF の手順を変えうる点 (Solver::line_points) から探す。
//! - 受ける側の四には攻める側が止めるしかない。止めたあとも脅威が残っていれば攻めは続く。
//! - 黒は禁じ手を打てず、禁じ手の点では止められない。黒の三は、達四になる点が禁じ手なら三ではない (rules.rs)。
//!
//! 「勝ちがある」と誤って判断しないよう、受ける側の手は漏らさず数える。読み切れなかった (局面数の上限に達した)
//! 局面も勝ちとしない。例外として、離れた石が黒の禁じ手の判定を変えるごくまれな形は受けとして数えない。
//!
//! 攻める側の手数 (四・三の数。受ける側の四を止める手は数えない) の上限を 1 手ずつ増やして読む (反復深化)。
//! 2 手目以降の攻めは、それまでの攻めの石の近くに限る (Solver::threat_moves)。そのため「VCT はない」という結果は
//! 「この読み方では見つからない」という意味で、見落としはありうる。

use std::collections::HashMap;

use crate::board::{Board, CELLS, GameResult, SIZE, Stone, pos};
use crate::vcf;

/// 読みの上限。
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// 1 回の探索で調べる局面数。超えたら読み切れなかったものとして扱う
    pub nodes: u32,
    /// 攻める側の手数 (最後の五の手と、受ける側の四を止める手は数えない)
    pub depth: u32,
    /// 2 手目以降の攻めを、それまでの攻めの石の近くに限るか (Solver::threat_moves)。
    /// 限らないと読む局面が大きく増えるので、読みの確認用
    pub focus: bool,
}

impl Default for Limits {
    fn default() -> Self {
        Self { nodes: 20_000, depth: 10, focus: true }
    }
}

/// 三の脅威 (VCF) を調べるときの VCF の局面数の上限。
const VCF_BUDGET: u32 = 2_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// VCT がある。値は最初の手
    Win(usize),
    /// Limits::depth 手以内の VCT はない
    NoWin,
    /// 局面数の上限に達して読み切れなかった
    Unknown,
}

impl Outcome {
    pub fn win(self) -> Option<usize> {
        match self {
            Outcome::Win(p) => Some(p),
            _ => None,
        }
    }
}

/// 手番側の VCT を探す。
pub fn solve(board: &Board, limits: Limits) -> Outcome {
    if board.result() != GameResult::Ongoing {
        return Outcome::NoWin;
    }
    let mut solver = Solver::new(board.side_to_move(), limits);
    if let Some(&p) = board.winning_moves(solver.attacker).first() {
        return Outcome::Win(p);
    }
    for depth in 1..=limits.depth {
        let Some((moves, next)) = solver.attacks(board, depth) else { return Outcome::NoWin };
        for p in moves {
            if solver.attack(board, p, next) {
                return Outcome::Win(p);
            }
            if solver.exhausted() {
                return Outcome::Unknown;
            }
        }
    }
    Outcome::NoWin
}

/// 手番側の VCT を探す。読み切れなかったときも None。
pub fn find(board: &Board, limits: Limits) -> Option<usize> {
    solve(board, limits).win()
}

/// 手番側が p に打ったあと、VCT で勝てるか (p 自身が VCT の最初の手か)。
pub fn wins_with(board: &Board, p: usize, limits: Limits) -> Outcome {
    if board.result() != GameResult::Ongoing {
        return Outcome::NoWin;
    }
    let mut solver = Solver::new(board.side_to_move(), limits);
    for depth in 0..limits.depth {
        if solver.attack(board, p, depth) {
            return Outcome::Win(p);
        }
        if solver.exhausted() {
            return Outcome::Unknown;
        }
    }
    Outcome::NoWin
}

/// 受ける側の手番の局面で、攻める側 (手番でない側) の三への受け: 打った後に攻める側の VCF がなくなる点と、
/// 受ける側の四。攻める側に四があるときや、受ける側が一手休んでも攻める側に VCF がないときは None。
pub fn defenses(board: &Board) -> Option<Vec<usize>> {
    let attacker = board.side_to_move().opponent();
    if !board.winning_moves(attacker).is_empty() {
        return None;
    }
    let mut solver = Solver::new(attacker, Limits { nodes: 0, ..Limits::default() });
    let mut passed = board.clone();
    passed.pass();
    let line = solver.vcf(&passed)?;
    Some(solver.defenses(board, &line))
}

/// 読み切った局面の結果
#[derive(Clone, Copy)]
enum Known {
    Win,
    /// この手数以下では勝てない
    NoWinWithin(u32),
}

struct Solver {
    attacker: Stone,
    budget: u32,
    focus: bool,
    nodes: u32,
    known: HashMap<u64, Known>,
    /// 攻める側の手番の局面で見つけた VCF (なければ None)
    vcfs: HashMap<u64, Option<Vec<usize>>>,
    /// 受ける側の手番の局面で、攻めを崩せた手
    refutations: HashMap<u64, usize>,
    /// 今読んでいる手順で攻める側が打った石の、4 方向の線上 4 路以内に何個あるか (related)
    near: [u8; CELLS],
}

impl Solver {
    fn new(attacker: Stone, limits: Limits) -> Self {
        Self {
            attacker,
            budget: limits.nodes,
            focus: limits.focus,
            nodes: 0,
            known: HashMap::new(),
            vcfs: HashMap::new(),
            refutations: HashMap::new(),
            near: [0; CELLS],
        }
    }

    fn exhausted(&self) -> bool {
        self.nodes > self.budget
    }

    /// 局面を 1 つ数える。上限を超えたら false。
    fn visit(&mut self) -> bool {
        self.nodes += 1;
        !self.exhausted()
    }

    fn lookup(&self, key: u64, depth: u32) -> Option<bool> {
        match self.known.get(&key)? {
            Known::Win => Some(true),
            &Known::NoWinWithin(d) if d >= depth => Some(false),
            _ => None,
        }
    }

    fn store(&mut self, key: u64, depth: u32, win: bool) {
        // 上限に達した後の結果は読み切れていない
        if !self.exhausted() {
            self.known.insert(key, if win { Known::Win } else { Known::NoWinWithin(depth) });
        }
    }

    /// 攻める側の手番の局面 board の VCF。同じ局面を何度も調べるので結果を覚えておく。
    fn vcf(&mut self, board: &Board) -> Option<Vec<usize>> {
        self.vcfs.entry(board.key()).or_insert_with(|| vcf::find(board, VCF_BUDGET)).clone()
    }

    /// 攻める側の手番で、読む手とその後の残り手数。受ける側の四があれば止める手だけ (手数は使わない)。
    /// 読む手がなければ None。
    fn attacks(&self, board: &Board, depth: u32) -> Option<(Vec<usize>, u32)> {
        match board.winning_moves(self.attacker.opponent()).as_slice() {
            [] if depth == 0 => None,
            [] => Some((self.threat_moves(board), depth - 1)),
            &[block] => Some((vec![block], depth)),
            _ => None,
        }
    }

    /// 攻める側が四か三を作る手。四を先に並べる。
    /// 最初の手のあとは、この手順で攻める側が打った石を通る線上 (4 路以内) の手だけを読む。勝ち筋は前の攻めの石を
    /// 使って続くことが多く、盤のどこででも三を作れるとすると読む局面が手数に対して急に増えるため。
    /// 攻める側の手を絞るだけなので、勝ちの判断は誤らないが、この外の手を使う VCT は見落とす。
    fn threat_moves(&self, board: &Board) -> Vec<usize> {
        let me = self.attacker;
        let mut candidates = board.four_candidates(me);
        candidates.extend(board.three_candidates(me));
        candidates.sort_unstable();
        candidates.dedup();
        if self.focus && self.near.iter().any(|&n| n > 0) {
            candidates.retain(|&p| self.near[p] > 0);
        }
        let (mut fours, mut threes) = (Vec::new(), Vec::new());
        for p in candidates {
            if me == Stone::Black && board.is_forbidden(p) {
                continue;
            }
            let mut after = board.clone();
            if !matches!(after.play(p), Ok(GameResult::Ongoing)) {
                continue;
            }
            if !after.winning_moves_through(me, p).is_empty() {
                fours.push(p);
            } else if after.makes_three(me, p) {
                threes.push(p);
            }
        }
        fours.extend(threes);
        fours
    }

    /// 攻める側の手番の局面。
    fn or_node(&mut self, board: &Board, depth: u32) -> bool {
        if !self.visit() {
            return false;
        }
        if !board.winning_moves(self.attacker).is_empty() {
            return true;
        }
        let key = board.key();
        if let Some(win) = self.lookup(key, depth) {
            return win;
        }
        let Some((moves, next)) = self.attacks(board, depth) else { return false };
        let win = moves.into_iter().any(|p| self.attack(board, p, next));
        self.store(key, depth, win);
        win
    }

    /// 攻める側が p に打ち、残り depth 手で勝てるか。
    fn attack(&mut self, board: &Board, p: usize, depth: u32) -> bool {
        if self.attacker == Stone::Black && board.is_forbidden(p) {
            return false;
        }
        let mut after = board.clone();
        match after.play(p) {
            Ok(GameResult::Win(winner)) => winner == self.attacker,
            Ok(GameResult::Ongoing) => {
                let related = near_lines(p);
                for &q in &related {
                    self.near[q] += 1;
                }
                let win = self.and_node(&after, depth);
                for &q in &related {
                    self.near[q] -= 1;
                }
                win
            }
            _ => false,
        }
    }

    /// 受ける側の手番の局面。受ける側のどの手に対しても攻める側が勝てるか。
    fn and_node(&mut self, board: &Board, depth: u32) -> bool {
        if !self.visit() {
            return false;
        }
        let defender = self.attacker.opponent();
        if !board.winning_moves(defender).is_empty() {
            return false;
        }
        match board.winning_moves(self.attacker).as_slice() {
            [] => {}
            &[block] => {
                if defender == Stone::Black && board.is_forbidden(block) {
                    return true;
                }
                let mut after = board.clone();
                return matches!(after.play(block), Ok(GameResult::Ongoing)) && self.or_node(&after, depth);
            }
            _ => return true,
        }

        let key = board.key();
        if let Some(win) = self.lookup(key, depth) {
            return win;
        }
        // 受ける側が一手休んでも攻める側に VCF がなければ、攻めは途切れている
        let mut passed = board.clone();
        passed.pass();
        let win = match self.vcf(&passed) {
            None => false,
            Some(line) => {
                let mut replies = self.defenses(board, &line);
                // 前の反復 (手数の上限が 1 つ少ない読み) で攻めを崩せた手を先に読む
                if let Some(i) = self.refutations.get(&key).and_then(|r| replies.iter().position(|q| q == r)) {
                    replies[..=i].rotate_right(1);
                }
                replies.into_iter().all(|q| {
                    let mut after = board.clone();
                    let attacker_wins = match after.play(q) {
                        Ok(GameResult::Ongoing) => self.or_node(&after, depth),
                        Ok(GameResult::Win(winner)) => winner == self.attacker,
                        _ => false,
                    };
                    if !attacker_wins {
                        self.refutations.insert(key, q);
                    }
                    attacker_wins
                })
            }
        };
        self.store(key, depth, win);
        win
    }

    /// 受ける側がそこに打つと VCF の手順 line が変わりうる点: 手順の点 (攻める側の手と受ける側の止める手)、
    /// 攻める側の四の五の点 (止める点以外も)、受ける側の止める手と合わせて受ける側の四になる点。
    /// これ以外の点に打っても、攻める側の四とその五の点、受ける側の石の並びは変わらないので、同じ手順がそのまま使える
    /// (黒の禁じ手の判定が離れた石で変わる、ごくまれな形を除く)。
    fn line_points(&self, board: &Board, line: &[usize]) -> Vec<usize> {
        let me = self.attacker;
        let defender = me.opponent();
        let mut points = Vec::new();
        // board は受ける側の手番。line は受ける側が一手休んだ局面での手順
        let mut b = board.clone();
        b.pass();
        for step in line.chunks(2) {
            let p = step[0];
            points.push(p);
            if !matches!(b.play(p), Ok(GameResult::Ongoing)) {
                break;
            }
            points.extend(b.winning_moves_through(me, p));
            let Some(&block) = step.get(1) else { break };
            points.push(block);
            if !matches!(b.play(block), Ok(GameResult::Ongoing)) {
                break;
            }
            // block を通る線上で、受ける側の石があと 1 つで四になる並びの空点
            let near = near_lines(block);
            points.extend(b.four_candidates(defender).into_iter().filter(|q| near.contains(q)));
        }
        points
    }

    /// 三 (攻める側の VCF の脅威) に対する受ける側の手: VCF を崩す点と、受ける側の四。line は見つけた VCF。
    fn defenses(&mut self, board: &Board, line: &[usize]) -> Vec<usize> {
        let defender = self.attacker.opponent();
        let playable = |q: usize| board.get(q).is_none() && !(defender == Stone::Black && board.is_forbidden(q));
        let mut seen = [false; CELLS];
        let mut fours = Vec::new();
        for q in board.four_candidates(defender) {
            if !playable(q) {
                continue;
            }
            let mut after = board.clone();
            if !matches!(after.play(q), Ok(GameResult::Ongoing)) {
                continue;
            }
            let fives = after.winning_moves_through(defender, q).len();
            if fives > 0 {
                seen[q] = true;
                fours.push((q, fives));
            }
        }

        let mut blocks = Vec::new();
        for q in self.line_points(board, line) {
            if std::mem::replace(&mut seen[q], true) || !playable(q) {
                continue;
            }
            let mut after = board.clone();
            if !matches!(after.play(q), Ok(GameResult::Ongoing)) {
                continue;
            }
            // 同じ手順がまだ使えるか、別の VCF があれば受けにならない
            if vcf::verify(&after, line) || self.vcf(&after).is_some() {
                continue;
            }
            blocks.push(q);
        }
        // 読む順: 止められない四 (活四・四四。攻める側が止めても受ける側が勝つ) → VCF を崩す点 (手順の前の点ほど
        // 攻めの要) → ほかの四。ほかの四は攻める側が止めたあとも同じ三が残り、受けになることが少ない
        let (open, other): (Vec<_>, Vec<_>) = fours.into_iter().partition(|&(_, fives)| fives >= 2);
        let mut replies: Vec<usize> = open.into_iter().map(|(q, _)| q).collect();
        replies.extend(blocks);
        replies.extend(other.into_iter().map(|(q, _)| q));
        replies
    }
}

/// s を通る 4 方向の線上で、s から 4 路以内の点。
fn near_lines(s: usize) -> Vec<usize> {
    let (row, col) = ((s / SIZE) as isize, (s % SIZE) as isize);
    let mut points = Vec::new();
    for (dr, dc) in [(0, 1), (1, 0), (1, 1), (1, -1)] {
        for k in (-4..=4).filter(|&k| k != 0) {
            let (r, c) = (row + dr * k, col + dc * k);
            if (0..SIZE as isize).contains(&r) && (0..SIZE as isize).contains(&c) {
                points.push(pos(r as usize, c as usize));
            }
        }
    }
    points
}
