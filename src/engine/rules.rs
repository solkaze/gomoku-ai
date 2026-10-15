//! 連珠の禁じ手・勝利判定。
//!
//! 参考: https://renjusha.net/about-renju/
//! - 黒は三三・四四・長連が禁じ手。ただし五ができる手は禁じ手より優先される。
//! - 三は「禁じ手にならない一手で達四になれる」ものだけを数える(再帰判定)。
//! - 白に禁じ手はなく、長連でも勝ち。
//!
//! 禁じ手は種類ごとに無効にできる ([`Rules`])。長連を無効にすると、黒も六以上の連で勝ちになる。

use crate::board::{CELLS, SIZE};

pub const EMPTY: u8 = 0;
pub const BLACK: u8 = 1;
pub const WHITE: u8 = 2;

/// 横・縦・右下がり・左下がり
const DIRS: [(isize, isize); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

type Cells = [u8; CELLS];

/// 黒に適用する禁じ手。学習 (自己対局・評価) は常に全て有効で行う。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules {
    pub double_three: bool,
    pub double_four: bool,
    pub overline: bool,
}

impl Default for Rules {
    fn default() -> Self {
        Self { double_three: true, double_four: true, overline: true }
    }
}

impl Rules {
    /// 禁じ手を全て無効にしたもの。
    pub const fn none() -> Self {
        Self { double_three: false, double_four: false, overline: false }
    }

    /// color の長さ len の連が五 (勝ち) か。長連が禁じ手の黒だけ、ちょうど五に限る。
    fn is_win(self, color: u8, len: isize) -> bool {
        if color == BLACK && self.overline { len == 5 } else { len >= 5 }
    }
}

/// p から方向 d に k 歩進んだ位置。盤外なら None。
fn step(p: usize, d: (isize, isize), k: isize) -> Option<usize> {
    let r = (p / SIZE) as isize + d.0 * k;
    let c = (p % SIZE) as isize + d.1 * k;
    let n = SIZE as isize;
    ((0..n).contains(&r) && (0..n).contains(&c)).then(|| (r * n + c) as usize)
}

/// p を含む color の連の両端を、p からの歩数 (lo <= 0 <= hi) で返す。
fn run(cells: &Cells, p: usize, d: (isize, isize), color: u8) -> (isize, isize) {
    let mut lo = 0;
    while step(p, d, lo - 1).is_some_and(|q| cells[q] == color) {
        lo -= 1;
    }
    let mut hi = 0;
    while step(p, d, hi + 1).is_some_and(|q| cells[q] == color) {
        hi += 1;
    }
    (lo, hi)
}

fn run_len(cells: &Cells, p: usize, d: (isize, isize), color: u8) -> isize {
    let (lo, hi) = run(cells, p, d, color);
    hi - lo + 1
}

/// p に置かれた石が五(長連が禁じ手の黒はちょうど五、それ以外は五以上)を作っているか。
pub fn is_five(cells: &Cells, p: usize, color: u8, rules: Rules) -> bool {
    DIRS.iter().any(|&d| rules.is_win(color, run_len(cells, p, d, color)))
}

/// 黒石が置かれた p について、方向 d 上で「p を含む五」を完成させる空点。
/// (個数, 最小オフセット, 最大オフセット) を返す。
fn five_points(cells: &mut Cells, p: usize, d: (isize, isize), rules: Rules) -> (u32, isize, isize) {
    let (mut n, mut min, mut max) = (0, isize::MAX, isize::MIN);
    for k in (-4..=4).filter(|&k| k != 0) {
        let Some(q) = step(p, d, k) else { continue };
        if cells[q] != EMPTY {
            continue;
        }
        cells[q] = BLACK;
        let (lo, hi) = run(cells, q, d, BLACK);
        cells[q] = EMPTY;
        // q から見た p の位置は -k
        if rules.is_win(BLACK, hi - lo + 1) && lo <= -k && -k <= hi {
            n += 1;
            min = min.min(k);
            max = max.max(k);
        }
    }
    (n, min, max)
}

/// 方向 d 上で p が作っている四の数。達四(・●●●●・)は 1 つと数え、
/// ●・●●●・● のような一直線の四四は 2 つと数える。
fn four_count(cells: &mut Cells, p: usize, d: (isize, isize), rules: Rules) -> u32 {
    match five_points(cells, p, d, rules) {
        (2, min, max) if max - min == 5 => 1,
        (n, _, _) => n,
    }
}

/// 方向 d 上で p が本物の三を作っているか:
/// p を含む達四を作れる空点があり、その点が禁じ手でないこと。
fn is_true_three(cells: &mut Cells, p: usize, d: (isize, isize), rules: Rules) -> bool {
    for k in (-4..=4).filter(|&k| k != 0) {
        let Some(q) = step(p, d, k) else { continue };
        if cells[q] != EMPTY {
            continue;
        }
        cells[q] = BLACK;
        let (n, min, max) = five_points(cells, q, d, rules);
        cells[q] = EMPTY;
        // 達四の両端の間に p があれば、その達四は p を含む
        let straight_four_with_p = n == 2 && max - min == 5 && min < -k && -k < max;
        if straight_four_with_p && !is_forbidden(cells, q, rules) {
            return true;
        }
    }
    false
}

/// 空点 p に黒が打つと禁じ手になるか。cells は判定中に一時的に書き換えるが、戻り値の時点で元に戻っている。
pub fn is_forbidden(cells: &mut Cells, p: usize, rules: Rules) -> bool {
    debug_assert_eq!(cells[p], EMPTY);
    // 禁じ手には p 以外に同一直線上(±4 以内)の黒石が最低 4 つ必要
    let nearby = DIRS
        .iter()
        .flat_map(|&d| (-4..=4).filter(|&k| k != 0).filter_map(move |k| step(p, d, k)))
        .filter(|&q| cells[q] == BLACK)
        .count();
    if nearby < 4 {
        return false;
    }

    cells[p] = BLACK;
    let forbidden = forbidden_with_stone(cells, p, rules);
    cells[p] = EMPTY;
    forbidden
}

fn forbidden_with_stone(cells: &mut Cells, p: usize, rules: Rules) -> bool {
    let lens = DIRS.map(|d| run_len(cells, p, d, BLACK));
    if lens.iter().any(|&len| rules.is_win(BLACK, len)) {
        return false;
    }
    if lens.iter().any(|&len| len > 5) {
        return true;
    }
    if !rules.double_three && !rules.double_four {
        return false;
    }

    let (mut fours, mut threes) = (0, 0);
    for d in DIRS {
        let f = four_count(cells, p, d, rules);
        if f > 0 {
            fours += f;
        } else if rules.double_three && is_true_three(cells, p, d, rules) {
            threes += 1;
        }
        if (rules.double_four && fours >= 2) || threes >= 2 {
            return true;
        }
    }
    false
}
