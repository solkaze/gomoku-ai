//! 自己対局 1 局分の生成。

use rand::Rng;

use crate::board::{Board, GameResult, Stone};
use crate::features::{FEATURE_LEN, encode};
use crate::mcts::{Evaluator, Mcts};

/// 学習データ 1 局面分。
pub struct Sample {
    pub features: Vec<u8>,
    pub policy: Vec<f32>,
    /// 手番側から見た最終結果
    pub value: f32,
}

pub struct Game {
    pub samples: Vec<Sample>,
    pub result: GameResult,
    pub moves: usize,
}

/// 1 局打つ。temperature_moves 手目までは訪問回数に比例して手を選び、以降は最多訪問手を打つ。
pub fn play_game<E: Evaluator, R: Rng>(
    mcts: &mut Mcts,
    evaluator: &mut E,
    temperature_moves: usize,
    rng: &mut R,
) -> Game {
    let mut board = Board::new();
    let mut records: Vec<(Vec<u8>, Vec<f32>, Stone)> = Vec::new();
    while board.result() == GameResult::Ongoing {
        let result = mcts.search(&board, evaluator, rng);
        // 黒の初手は天元で固定なので学習しない
        if board.move_count() > 0 {
            let mut features = vec![0; FEATURE_LEN];
            encode(&board, &mut features);
            records.push((features, result.policy(), board.side_to_move()));
        }
        let temperature = if board.move_count() < temperature_moves { 1.0 } else { 0.0 };
        board.play(result.sample_move(temperature, rng)).expect("探索結果は合法手");
    }

    let outcome = board.result();
    let samples = records
        .into_iter()
        .map(|(features, policy, side)| {
            let value = match outcome {
                GameResult::Win(winner) if winner == side => 1.0,
                GameResult::Win(_) => -1.0,
                _ => 0.0,
            };
            Sample { features, policy, value }
        })
        .collect();
    Game { samples, result: outcome, moves: board.move_count() }
}

/// 別々の評価器 (モデル) 同士で 1 局打ち、結果を返す。強さの評価用。
/// temperature_moves 手目までは訪問回数に比例して手を選び、同じ棋譜ばかりにならないようにする。
pub fn play_match<R: Rng>(
    mcts: &mut Mcts,
    black: &mut dyn Evaluator,
    white: &mut dyn Evaluator,
    temperature_moves: usize,
    rng: &mut R,
) -> GameResult {
    let mut board = Board::new();
    while board.result() == GameResult::Ongoing {
        let evaluator: &mut dyn Evaluator = match board.side_to_move() {
            Stone::Black => &mut *black,
            Stone::White => &mut *white,
        };
        let result = mcts.search(&board, evaluator, rng);
        let temperature = if board.move_count() < temperature_moves { 1.0 } else { 0.0 };
        board.play(result.sample_move(temperature, rng)).expect("探索結果は合法手");
    }
    board.result()
}
