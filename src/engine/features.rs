//! ネットワーク入力の特徴平面。src/train/network.py と同じ定義にすること。
//!
//! 0: 手番側の石 / 1: 相手の石 / 2: 手番が黒なら全て 1 / 3: 黒の禁じ手の点
//! 4: 手番側の VCF の攻め手 (見つけた手順で手番側が打つ点。VCF がなければ全て 0)
//! 5: 相手の VCF の攻め手 (手番側が一手休んだとして読む。相手に今ある脅威)
//!
//! 4・5 は、長い強制手順をネットワークが苦手とする問題 (囲碁のシチョウと同じ) を補うため、読みの結果を入力に渡す
//! もの (AlphaGo・KataGo のシチョウ特徴にならった)。0〜3 だけを入力にする古いモデル (BASE_PLANES) も読める。

use crate::board::{Board, CELLS, Stone};
use crate::vcf;

pub const PLANES: usize = 6;
pub const FEATURE_LEN: usize = PLANES * CELLS;
/// VCF の平面を加える前のモデルの入力平面数
pub const BASE_PLANES: usize = 4;
/// 特徴の VCF を読むときの局面数の上限。学習データと対局で同じ値を使うこと (変えると入力の意味が変わる)。
/// 自己対局の局面では、2000 にしても見つかる VCF は 1% ほどしか増えず、戦術的な局面での時間が大きく増えた
/// (1 手 16000 回の探索で、上限 2000 では特徴なしの約 2 倍、200 では 1.2 倍ほどの時間)
pub const VCF_BUDGET: u32 = 200;

/// board の特徴を out に書く。out の長さで平面数 (BASE_PLANES か PLANES) を決め、BASE_PLANES なら VCF は読まない。
pub fn encode(board: &Board, out: &mut [u8]) {
    let planes = out.len() / CELLS;
    assert!(out.len() % CELLS == 0 && (planes == BASE_PLANES || planes == PLANES), "平面数 {planes} には対応していない");
    let side = board.side_to_move();
    let forbidden = board.forbidden_mask();
    let (base, vcf_planes) = out.split_at_mut(BASE_PLANES * CELLS);
    let (own, rest) = base.split_at_mut(CELLS);
    let (opp, rest) = rest.split_at_mut(CELLS);
    let (black_to_move, forbidden_plane) = rest.split_at_mut(CELLS);
    for p in 0..CELLS {
        let stone = board.get(p);
        own[p] = (stone == Some(side)) as u8;
        opp[p] = (stone == Some(side.opponent())) as u8;
        black_to_move[p] = (side == Stone::Black) as u8;
        forbidden_plane[p] = forbidden[p] as u8;
    }
    if planes == PLANES {
        let (own_vcf, opp_vcf) = vcf_planes.split_at_mut(CELLS);
        mark_vcf(board, own_vcf);
        let mut passed = board.clone();
        passed.pass();
        mark_vcf(&passed, opp_vcf);
    }
}

/// board の手番側の VCF の攻め手を plane に 1 で書く (手順は攻め手と受けの手が交互に並ぶ)。
fn mark_vcf(board: &Board, plane: &mut [u8]) {
    plane.fill(0);
    if let Some(line) = vcf::find(board, VCF_BUDGET) {
        for &p in line.iter().step_by(2) {
            plane[p] = 1;
        }
    }
}
