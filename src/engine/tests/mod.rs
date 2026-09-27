use crate::{Board, Stone, pos};

mod config;
mod mcts;
mod selfplay;
mod rules;

fn board(black: &[(usize, usize)], white: &[(usize, usize)], side: Stone) -> Board {
    let to_pos = |s: &[(usize, usize)]| s.iter().map(|&(r, c)| pos(r, c)).collect::<Vec<_>>();
    Board::from_stones(&to_pos(black), &to_pos(white), side)
}

fn black_to_move(black: &[(usize, usize)], white: &[(usize, usize)]) -> Board {
    board(black, white, Stone::Black)
}
