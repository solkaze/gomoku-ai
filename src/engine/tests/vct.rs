use super::board;
use super::vcf::after;
use crate::vct::{Limits, Outcome, solve, wins_with};
use crate::{Stone, pos, vcf};

/// (7,5) (7,6) と (5,7) (6,7) の 2 つの二。(7,7) に打つと横と縦の三が同時にできる
const TWO_TWOS: [(usize, usize); 4] = [(7, 5), (7, 6), (5, 7), (6, 7)];

/// TWO_TWOS の斜めの並び ((7,5)-(5,7) と (7,6)-(6,7)) を塞ぐ相手の石。残る狙いは (7,7) の三三だけになる
const BLOCKERS: [(usize, usize); 3] = [(6, 6), (8, 5), (5, 8)];

#[test]
fn white_wins_with_a_double_three() {
    // 白に禁じ手はないので、(7,7) の三三から勝てる
    let board = board(&BLOCKERS, &TWO_TWOS, Stone::White);
    assert_eq!(vcf::find(&board, vcf::DEFAULT_BUDGET), None);
    assert!(matches!(solve(&board, Limits::default()), Outcome::Win(_)));
    assert_eq!(wins_with(&board, pos(7, 7), Limits::default()), Outcome::Win(pos(7, 7)));
}

#[test]
fn black_cannot_win_with_a_forbidden_double_three() {
    // 同じ形でも黒の (7,7) は三三の禁じ手。他の三には (7,7) で受けられる
    let board = board(&TWO_TWOS, &BLOCKERS, Stone::Black);
    assert!(board.is_forbidden(pos(7, 7)));
    assert_eq!(solve(&board, Limits::default()), Outcome::NoWin);
    assert_eq!(wins_with(&board, pos(7, 7), Limits::default()), Outcome::NoWin);
}

#[test]
fn black_wins_with_a_four_three() {
    // 黒の (7,7) は横の四 (左端は白で塞がれている) と縦の三。四三は禁じ手ではない
    let black = [(7, 4), (7, 5), (7, 6), (5, 7), (6, 7)];
    let white = [(7, 3), (6, 6), (8, 5), (5, 8)];
    let board = board(&black, &white, Stone::Black);
    assert!(!board.is_forbidden(pos(7, 7)));
    // 四三は、四を止めさせてから三を達四にするので VCF でもある
    assert!(vcf::find(&board, vcf::DEFAULT_BUDGET).is_some());
    assert!(matches!(solve(&board, Limits::default()), Outcome::Win(_)));
    assert_eq!(wins_with(&board, pos(7, 7), Limits::default()), Outcome::Win(pos(7, 7)));
    // 四を先に打って (7,7) で止められると勝てない (読み切れないこともあるが、勝ちとは判断しない)
    assert_ne!(wins_with(&board, pos(7, 8), Limits::default()), Outcome::Win(pos(7, 8)));
}

#[test]
fn an_open_three_of_the_defender_refutes_threes() {
    // 白の三三の形があっても、黒に活三があれば、白の三には黒の達四で応じられる
    let mut black = BLOCKERS.to_vec();
    black.extend([(2, 10), (2, 11), (2, 12)]);
    let board = board(&black, &TWO_TWOS, Stone::White);
    assert_eq!(solve(&board, Limits::default()), Outcome::NoWin);
    assert_eq!(wins_with(&board, pos(7, 7), Limits::default()), Outcome::NoWin);
}

#[test]
fn a_vcf_is_also_a_vct() {
    // 実戦の 41 手目 (黒番) には VCF がある
    let board = after(40);
    let line = vcf::find(&board, vcf::DEFAULT_BUDGET).unwrap();
    assert!(matches!(solve(&board, Limits::default()), Outcome::Win(_)));
    assert_eq!(wins_with(&board, line[0], Limits::default()), Outcome::Win(line[0]));
}

#[test]
fn vcf_verify_checks_the_line_on_a_changed_board() {
    let board = after(40);
    let line = vcf::find(&board, vcf::DEFAULT_BUDGET).unwrap();
    assert!(vcf::verify(&board, &line));
    // 白が最初の四の点に先に打っていれば、同じ手順は使えない
    let mut blocked = board.clone();
    blocked.pass();
    blocked.play(line[0]).unwrap();
    assert!(!vcf::verify(&blocked, &line));
}
