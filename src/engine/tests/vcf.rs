use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::vcf::{DEFAULT_BUDGET, find, safe_moves};
use crate::{Board, GameResult, Mcts, MctsConfig, SIZE, Stone, UniformEvaluator, pos};

/// 人 (黒) が AI (白) に勝った実戦の棋譜 (x y)。39 手目 (6 9) のあと、白の 40 手目 (7 4) が黒の VCF を
/// 防げず、41 手目から黒の VCF で終わった。
const GAME: [(usize, usize); 47] = [
    (7, 7), (6, 6), (6, 7), (8, 7), (5, 7), (4, 7), (5, 8), (5, 9), (7, 6), (8, 5),
    (7, 8), (7, 9), (4, 9), (3, 10), (3, 8), (4, 8), (8, 9), (9, 10), (6, 10), (9, 8),
    (9, 11), (5, 10), (8, 11), (7, 11), (4, 5), (5, 6), (3, 5), (3, 6), (2, 5), (1, 5),
    (2, 6), (4, 4), (8, 12), (8, 10), (10, 10), (7, 13), (9, 9), (8, 8), (6, 9), (7, 4),
    (6, 8), (6, 11), (2, 4), (4, 6), (2, 3), (2, 7), (2, 2),
];

pub(super) fn xy(x: usize, y: usize) -> usize {
    pos(y, x)
}

/// 実戦の n 手目までを打った局面
pub(super) fn after(n: usize) -> Board {
    let mut board = Board::new();
    for &(x, y) in &GAME[..n] {
        board.play(xy(x, y)).unwrap();
    }
    board
}

/// line を打ち進めて、手番側 (line の最初の手の側) が五で勝つことを確かめる。
fn assert_wins(board: &Board, line: &[usize]) {
    let me = board.side_to_move();
    let mut b = board.clone();
    for &p in line {
        let result = b.play(p).unwrap();
        if result != GameResult::Ongoing {
            assert_eq!(result, GameResult::Win(me));
            return;
        }
    }
    // 活四などで終わる手順は相手の受けまでで止まるので、手番側にまだ五の点が残っている
    assert_eq!(b.side_to_move(), me);
    assert!(!b.winning_moves(me).is_empty());
}

#[test]
fn finds_the_winning_sequence_of_fours_from_the_game() {
    // 41 手目 (黒番): 実戦では (6 8) から VCF で勝った
    let board = after(40);
    let line = find(&board, DEFAULT_BUDGET).expect("黒に VCF がある");
    assert_wins(&board, &line);
    // 白番にも VCF はない
    assert_eq!(find(&after(39), DEFAULT_BUDGET), None);
}

#[test]
fn no_vcf_in_a_quiet_position() {
    assert_eq!(find(&after(10), DEFAULT_BUDGET), None);
}

#[test]
fn safe_moves_are_the_moves_that_stop_the_opponents_vcf() {
    // 40 手目 (白番): 黒の VCF を防げるのは 5 点だけで、実戦の (7 4) はその中にない
    let board = after(39);
    let safe = safe_moves(&board, &board.legal_moves(), DEFAULT_BUDGET);
    let mut expected: Vec<usize> = [(8, 4), (8, 6), (6, 8), (7, 10), (6, 11)].iter().map(|&(x, y)| xy(x, y)).collect();
    expected.sort();
    assert_eq!(safe, expected);
    assert!(!safe.contains(&xy(7, 4)));
}

#[test]
fn black_cannot_block_a_four_on_a_forbidden_point() {
    // 黒: (7,5) (7,6) と (5,7) (6,7) があり (7,7) が三三の禁点。(2,2) も黒で、白の斜めの並びの片側を塞ぐ。
    // 白が (6,6) に打つと四の五の点は (7,7) だけになり、黒はそこで止めると禁じ手で負ける
    let black = [(7, 5), (7, 6), (5, 7), (6, 7), (2, 2)].map(|(r, c)| pos(r, c));
    let white = [(3, 3), (4, 4), (5, 5), (14, 14)].map(|(r, c)| pos(r, c));
    let mut board = Board::from_stones(&black, &white, Stone::White);
    assert!(board.is_forbidden(pos(7, 7)));
    assert_eq!(find(&board, DEFAULT_BUDGET), Some(vec![pos(6, 6), pos(7, 7)]));

    // 禁じ手がなければ (7,7) で止められるので、同じ手順にはならない
    board.set_rules(crate::Rules::none());
    assert_ne!(find(&board, DEFAULT_BUDGET), Some(vec![pos(6, 6), pos(7, 7)]));
}

#[test]
fn mcts_with_vcf_keeps_only_moves_that_stop_the_vcf() {
    let board = after(39);
    let mut rng = StdRng::seed_from_u64(0);
    let mut mcts = Mcts::new(MctsConfig { simulations: 64, vcf: true, ..Default::default() });
    let result = mcts.search(&board, &mut UniformEvaluator, &mut rng);
    let mut moves: Vec<usize> = result.visits.iter().map(|&(mv, _)| mv).collect();
    moves.sort();
    assert_eq!(moves.len(), 5);
    assert!(!moves.contains(&xy(7, 4)));

    // VCF のある黒番では、その最初の手だけを読む
    let board = after(40);
    let result = mcts.search(&board, &mut UniformEvaluator, &mut rng);
    assert_eq!(result.visits.len(), 1);
    let line = find(&board, DEFAULT_BUDGET).unwrap();
    assert_eq!(result.best_move(), line[0]);

    // vcf を切れば絞らない
    let mut plain = Mcts::new(MctsConfig { simulations: 64, ..Default::default() });
    assert!(plain.search(&after(39), &mut UniformEvaluator, &mut rng).visits.len() > 100);
    let _ = SIZE;
}
