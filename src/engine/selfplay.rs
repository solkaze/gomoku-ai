//! 自己対局 1 局分の生成。

use rand::Rng;

use crate::board::{Board, CELLS, CENTER, GameResult, SIZE, Stone};
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

/// 開局のランダムな着手は、既にある石からこの距離 (チェビシェフ距離) 以内に打つ。
const OPENING_REACH: usize = 2;

/// 開局をランダムにした局面を作る。黒の初手 (天元) のあとに 0〜max_moves 手 (局ごとに一様に選ぶ) を
/// ランダムに打つ。0 手のときは通常の初期局面を返すので、普通の開局も一定の割合で残る。
/// 黒の禁じ手は打たず、決着のついた局面は返さない。
pub fn random_opening<R: Rng>(max_moves: usize, rng: &mut R) -> Board {
    loop {
        let moves = rng.random_range(0..=max_moves);
        let mut board = Board::new();
        if moves == 0 {
            return board;
        }
        board.play(CENTER).expect("天元は初手として合法");
        for _ in 0..moves {
            let candidates = opening_candidates(&board);
            if candidates.is_empty() {
                break;
            }
            board.play(candidates[rng.random_range(0..candidates.len())]).expect("候補は合法手");
            if board.result() != GameResult::Ongoing {
                break;
            }
        }
        // ランダムな手で五ができてしまったら作り直す
        if board.result() == GameResult::Ongoing {
            return board;
        }
    }
}

/// 石の近くの空点。黒番では禁じ手を除く。
fn opening_candidates(board: &Board) -> Vec<usize> {
    let stones: Vec<(usize, usize)> =
        (0..CELLS).filter(|&p| board.get(p).is_some()).map(|p| (p / SIZE, p % SIZE)).collect();
    let near = |p: usize| {
        let (row, col) = (p / SIZE, p % SIZE);
        stones.iter().any(|&(r, c)| row.abs_diff(r) <= OPENING_REACH && col.abs_diff(c) <= OPENING_REACH)
    };
    let black = board.side_to_move() == Stone::Black;
    (0..CELLS).filter(|&p| board.is_legal(p) && near(p) && !(black && board.is_forbidden(p))).collect()
}

/// opening の局面から 1 局打つ。そこから temperature_moves 手は訪問回数に比例して手を選び、以降は最多訪問手を打つ。
/// 学習データにするのは探索して打った局面だけで、opening までの手は含めない。
pub fn play_game<E: Evaluator, R: Rng>(
    mcts: &mut Mcts,
    evaluator: &mut E,
    opening: &Board,
    temperature_moves: usize,
    rng: &mut R,
) -> Game {
    let mut board = opening.clone();
    let temperature_until = board.move_count() + temperature_moves;
    let mut records: Vec<(Vec<u8>, Vec<f32>, Stone)> = Vec::new();
    while board.result() == GameResult::Ongoing {
        let result = mcts.search(&board, evaluator, rng);
        // 黒の初手は天元で固定なので学習しない
        if board.move_count() > 0 {
            let mut features = vec![0; FEATURE_LEN];
            encode(&board, &mut features);
            records.push((features, result.policy(), board.side_to_move()));
        }
        let temperature = if board.move_count() < temperature_until { 1.0 } else { 0.0 };
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

/// 別々の評価器 (モデル) 同士で opening の局面から 1 局打ち、結果を返す。強さの評価用。
/// そこから temperature_moves 手は訪問回数に比例して手を選び、同じ棋譜ばかりにならないようにする。
pub fn play_match<R: Rng>(
    mcts: &mut Mcts,
    black: &mut dyn Evaluator,
    white: &mut dyn Evaluator,
    opening: &Board,
    temperature_moves: usize,
    rng: &mut R,
) -> GameResult {
    let mut board = opening.clone();
    let temperature_until = board.move_count() + temperature_moves;
    while board.result() == GameResult::Ongoing {
        let evaluator: &mut dyn Evaluator = match board.side_to_move() {
            Stone::Black => &mut *black,
            Stone::White => &mut *white,
        };
        let result = mcts.search(&board, evaluator, rng);
        let temperature = if board.move_count() < temperature_until { 1.0 } else { 0.0 };
        board.play(result.sample_move(temperature, rng)).expect("探索結果は合法手");
    }
    board.result()
}
