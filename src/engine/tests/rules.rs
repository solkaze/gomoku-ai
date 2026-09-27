use super::{black_to_move, board};
use crate::{Board, CENTER, GameResult, MoveError, Stone, pos};

#[test]
fn first_move_must_be_center() {
    let mut b = Board::new();
    assert_eq!(b.legal_moves(), vec![CENTER]);
    assert_eq!(b.play(pos(0, 0)), Err(MoveError::NotCenter));
    assert_eq!(b.play(CENTER), Ok(GameResult::Ongoing));
    assert_eq!(b.side_to_move(), Stone::White);
    assert_eq!(b.play(CENTER), Err(MoveError::Occupied));
}

#[test]
fn double_three_is_forbidden() {
    // 横 (7,5)(7,6)+p と 縦 (5,7)(6,7)+p の活三が 2 つ
    let b = black_to_move(&[(7, 5), (7, 6), (5, 7), (6, 7)], &[]);
    assert!(b.is_forbidden(pos(7, 7)));
    assert_eq!(b.clone().play(pos(7, 7)), Err(MoveError::Forbidden));
}

#[test]
fn split_three_counts_as_three() {
    // 横 ●・●p の飛び三 + 縦の活三
    let b = black_to_move(&[(7, 4), (7, 6), (5, 7), (6, 7)], &[]);
    assert!(b.is_forbidden(pos(7, 7)));
}

#[test]
fn blocked_three_is_not_three() {
    // 横は ○●●p の止め三なので三三ではない
    let b = black_to_move(&[(7, 5), (7, 6), (5, 7), (6, 7)], &[(7, 4)]);
    assert!(!b.is_forbidden(pos(7, 7)));
}

#[test]
fn four_three_is_allowed() {
    let b = black_to_move(&[(7, 4), (7, 5), (7, 6), (5, 7), (6, 7)], &[]);
    assert!(!b.is_forbidden(pos(7, 7)));
}

#[test]
fn double_four_is_forbidden() {
    // 横は止め四、縦も四
    let b = black_to_move(&[(7, 4), (7, 5), (7, 6), (4, 7), (5, 7), (6, 7)], &[(7, 3), (3, 7)]);
    assert!(b.is_forbidden(pos(7, 7)));
}

#[test]
fn same_line_double_four_is_forbidden() {
    // ●・●p●・● : (7,4) と (7,8) のどちらでも五になる一直線の四四
    let b = black_to_move(&[(7, 3), (7, 5), (7, 7), (7, 9)], &[]);
    assert!(b.is_forbidden(pos(7, 6)));
}

#[test]
fn straight_four_is_single_four() {
    // ・●●●p・ の達四は四 1 つなので禁じ手ではない
    let b = black_to_move(&[(7, 4), (7, 5), (7, 6)], &[]);
    assert!(!b.is_forbidden(pos(7, 7)));
}

#[test]
fn overline_is_forbidden() {
    let b = black_to_move(&[(7, 2), (7, 3), (7, 4), (7, 6), (7, 7)], &[]);
    assert!(b.is_forbidden(pos(7, 5)));
    let b = black_to_move(&[(7, 2), (7, 3), (7, 4), (7, 6), (7, 7), (7, 8)], &[]);
    assert!(b.is_forbidden(pos(7, 5)));
}

#[test]
fn five_takes_precedence_over_forbidden() {
    // 横で五、同時に縦と斜めで四四になるが五が優先
    let mut b = black_to_move(
        &[(7, 3), (7, 4), (7, 5), (7, 6), (4, 7), (5, 7), (6, 7), (4, 4), (5, 5), (6, 6)],
        &[],
    );
    assert!(!b.is_forbidden(pos(7, 7)));
    assert_eq!(b.play(pos(7, 7)), Ok(GameResult::Win(Stone::Black)));
}

#[test]
fn three_that_can_only_become_overline_is_not_three() {
    // 横: ●・・●●p・・● 達四を作ると両側どちらも長連にしかならないので三ではない
    let b = black_to_move(&[(7, 0), (7, 3), (7, 4), (7, 8), (5, 5), (6, 5)], &[]);
    assert!(!b.is_forbidden(pos(7, 5)));
    // 端の黒石がなければ三三
    let b = black_to_move(&[(7, 3), (7, 4), (5, 5), (6, 5)], &[]);
    assert!(b.is_forbidden(pos(7, 5)));
}

#[test]
fn three_whose_straight_four_point_is_forbidden_is_not_three() {
    // 横: ○・●●p・ の三は (7,8) でしか達四にならない。
    // (7,8) は縦 (5,8)(6,8) と斜め (6,9)(5,10) で三三になる禁点なので、横は三ではない。
    let black = [(7, 5), (7, 6), (5, 7), (6, 7), (5, 8), (6, 8), (6, 9), (5, 10)];
    let b = black_to_move(&black, &[(7, 3)]);
    let with_p = black_to_move(&[black.as_slice(), &[(7, 7)]].concat(), &[(7, 3)]);
    assert!(with_p.is_forbidden(pos(7, 8)));
    assert!(!b.is_forbidden(pos(7, 7)));

    // (7,8) を禁点にしている石がなければ三三
    let b = black_to_move(&[(7, 5), (7, 6), (5, 7), (6, 7)], &[(7, 3)]);
    assert!(b.is_forbidden(pos(7, 7)));
}

#[test]
fn white_has_no_restrictions_and_overline_wins() {
    let mut b = board(
        &[(0, 0), (0, 2), (0, 4), (0, 6), (0, 8), (0, 10)],
        &[(7, 2), (7, 3), (7, 4), (7, 6), (7, 7)],
        Stone::White,
    );
    assert!(b.is_legal(pos(7, 5)));
    assert_eq!(b.play(pos(7, 5)), Ok(GameResult::Win(Stone::White)));
    assert_eq!(b.play(pos(1, 1)), Err(MoveError::GameOver));
}

#[test]
fn white_double_three_is_allowed() {
    let b = board(&[(0, 0), (0, 2), (0, 4), (0, 6)], &[(7, 5), (7, 6), (5, 7), (6, 7)], Stone::White);
    assert!(b.is_legal(pos(7, 7)));
}

#[test]
fn black_exact_five_wins() {
    let mut b = board(&[(7, 3), (7, 4), (7, 5), (7, 6)], &[(0, 0), (0, 2), (0, 4)], Stone::Black);
    assert_eq!(b.play(pos(7, 7)), Ok(GameResult::Win(Stone::Black)));
}

#[test]
fn legal_moves_exclude_forbidden_points() {
    let b = black_to_move(&[(7, 5), (7, 6), (5, 7), (6, 7)], &[]);
    let moves = b.legal_moves();
    assert!(!moves.contains(&pos(7, 7)));
    assert_eq!(moves.len(), 225 - 4 - 1);
}
