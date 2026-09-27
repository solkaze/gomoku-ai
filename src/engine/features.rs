//! ネットワーク入力の特徴平面。src/train/network.py と同じ定義にすること。
//!
//! 0: 手番側の石 / 1: 相手の石 / 2: 手番が黒なら全て 1 / 3: 黒の禁じ手の点

use crate::board::{Board, CELLS, Stone};

pub const PLANES: usize = 4;
pub const FEATURE_LEN: usize = PLANES * CELLS;

pub fn encode(board: &Board, out: &mut [u8]) {
    assert_eq!(out.len(), FEATURE_LEN);
    let side = board.side_to_move();
    let forbidden = board.forbidden_mask();
    let (own, rest) = out.split_at_mut(CELLS);
    let (opp, rest) = rest.split_at_mut(CELLS);
    let (black_to_move, forbidden_plane) = rest.split_at_mut(CELLS);
    for p in 0..CELLS {
        let stone = board.get(p);
        own[p] = (stone == Some(side)) as u8;
        opp[p] = (stone == Some(side.opponent())) as u8;
        black_to_move[p] = (side == Stone::Black) as u8;
        forbidden_plane[p] = forbidden[p] as u8;
    }
}
