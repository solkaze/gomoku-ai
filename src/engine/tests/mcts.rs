use rand::SeedableRng;
use rand::rngs::StdRng;

use super::{black_to_move, board};
use crate::{
    Board, CENTER, Evaluation, Evaluator, Mcts, MctsConfig, SearchResult, Stone, UniformEvaluator, pos,
};

/// 呼び出しのバッチサイズを記録する一様評価器
#[derive(Default)]
struct CountingEvaluator {
    batches: Vec<usize>,
}

impl Evaluator for CountingEvaluator {
    fn evaluate(&mut self, boards: &[Board]) -> Vec<Evaluation> {
        self.batches.push(boards.len());
        UniformEvaluator.evaluate(boards)
    }
}

fn search(board: &Board, config: MctsConfig) -> SearchResult {
    let mut rng = StdRng::seed_from_u64(0);
    Mcts::new(config).search(board, &mut UniformEvaluator, &mut rng)
}

#[test]
fn first_move_is_center_only() {
    let result = search(&Board::new(), MctsConfig { simulations: 32, ..Default::default() });
    assert_eq!(result.visits, vec![(CENTER, 32)]);
}

#[test]
fn finds_immediate_win() {
    let b = black_to_move(&[(7, 3), (7, 4), (7, 5), (7, 6)], &[(0, 0), (0, 2), (0, 4), (14, 14)]);
    let result = search(&b, MctsConfig::default());
    assert!([pos(7, 2), pos(7, 7)].contains(&result.best_move()));
    assert!(result.value > 0.9, "value = {}", result.value);
}

#[test]
fn forbidden_moves_are_not_searched() {
    let b = black_to_move(&[(7, 5), (7, 6), (5, 7), (6, 7)], &[(0, 0), (0, 2), (0, 4), (14, 14)]);
    let result = search(&b, MctsConfig { simulations: 64, ..Default::default() });
    assert!(result.visits.iter().all(|&(mv, _)| mv != pos(7, 7)));
}

#[test]
fn visits_match_simulations_and_batches_are_bounded() {
    let b = board(&[(7, 7), (6, 8)], &[(7, 8)], Stone::White);
    let config = MctsConfig { simulations: 200, batch_size: 8, ..Default::default() };
    let mut evaluator = CountingEvaluator::default();
    let mut rng = StdRng::seed_from_u64(0);
    let result = Mcts::new(config).search(&b, &mut evaluator, &mut rng);

    assert_eq!(result.visits.iter().map(|&(_, n)| n).sum::<u32>(), 200);
    assert!(evaluator.batches.iter().all(|&n| (1..=8).contains(&n)));
    let policy = result.policy();
    assert!((policy.iter().sum::<f32>() - 1.0).abs() < 1e-4);
}

#[test]
fn root_noise_keeps_search_valid() {
    let b = board(&[(7, 7)], &[], Stone::White);
    let config = MctsConfig { simulations: 100, dirichlet_epsilon: 0.25, ..Default::default() };
    let result = search(&b, config);
    assert_eq!(result.visits.iter().map(|&(_, n)| n).sum::<u32>(), 100);
    let mut rng = StdRng::seed_from_u64(1);
    let mv = result.sample_move(1.0, &mut rng);
    assert!(b.is_legal(mv));
}

#[test]
fn blocks_opponent_four() {
    let b = board(&[(7, 3), (7, 4), (7, 5), (7, 6), (0, 0)], &[(7, 2), (0, 2), (0, 4), (14, 14)], Stone::White);
    let result = search(&b, MctsConfig { simulations: 64, ..Default::default() });
    assert_eq!(result.visits, vec![(pos(7, 7), 64)]);
}

#[test]
fn black_cannot_block_on_forbidden_point() {
    // 白の四を止める (7,7) は黒の三三(横と斜め)なので、黒は他の合法手から選ぶしかない
    let b = black_to_move(
        &[(7, 5), (7, 6), (5, 5), (6, 6), (12, 7)],
        &[(8, 7), (9, 7), (10, 7), (11, 7), (14, 14)],
    );
    assert!(b.is_forbidden(pos(7, 7)));
    let result = search(&b, MctsConfig { simulations: 64, ..Default::default() });
    assert!(result.visits.len() > 1);
    assert!(result.visits.iter().all(|&(mv, _)| mv != pos(7, 7)));
}

#[test]
fn black_plays_forbidden_point_only_when_nothing_else_is_left() {
    // 空点は (7,5) だけで、そこは長連。打って負けるしかない
    let black_stones = [(7, 2), (7, 3), (7, 4), (7, 6), (7, 7)];
    let white_stones: Vec<(usize, usize)> = (0..15)
        .flat_map(|r| (0..15).map(move |c| (r, c)))
        .filter(|&rc| rc != (7, 5) && !black_stones.contains(&rc))
        .collect();
    let b = black_to_move(&black_stones, &white_stones);
    assert!(b.is_forbidden(pos(7, 5)));
    let result = search(&b, MctsConfig { simulations: 16, ..Default::default() });
    assert_eq!(result.visits, vec![(pos(7, 5), 16)]);
    assert!(result.value < -0.9, "value = {}", result.value);
}

/// cargo test --release --lib search_speed -- --ignored --nocapture
#[test]
#[ignore]
fn search_speed() {
    let mut b = Board::new();
    let mut rng = StdRng::seed_from_u64(0);
    let mut mcts = Mcts::new(MctsConfig::default());
    let start = std::time::Instant::now();
    let mut sims = 0;
    while b.move_count() < 30 && b.result() == crate::GameResult::Ongoing {
        let result = mcts.search(&b, &mut UniformEvaluator, &mut rng);
        sims += mcts.config().simulations;
        b.play(result.sample_move(1.0, &mut rng)).unwrap();
    }
    let secs = start.elapsed().as_secs_f64();
    println!("{} moves, {sims} sims in {secs:.2}s = {:.0} sims/s", b.move_count(), sims as f64 / secs);
}

/// 指定した点の方策を高くし、価値は白の負けとする評価器
struct LosingWhiteEvaluator {
    focus: Vec<usize>,
}

impl Evaluator for LosingWhiteEvaluator {
    fn evaluate(&mut self, boards: &[Board]) -> Vec<Evaluation> {
        boards
            .iter()
            .map(|b| {
                let mut policy = vec![0.0; crate::CELLS];
                for &p in &self.focus {
                    policy[p] = 8.0;
                }
                let value = if b.side_to_move() == Stone::White { -1.0 } else { 1.0 };
                Evaluation { policy, value }
            })
            .collect()
    }
}

#[test]
fn losing_side_prolongs_the_game() {
    // 黒には活三が 2 つあり、白はどちらかしか止められない。白は (0,3) か (0,4) で四を打てば、黒に止めさせて
    // 負けを 2 手遠ざけられる。評価はどの手も -1 なので、補正がなければ四を選ぶ理由がない
    let b = board(
        &[(7, 5), (7, 6), (7, 7), (9, 12), (10, 12), (11, 12)],
        &[(0, 0), (0, 1), (0, 2)],
        Stone::White,
    );
    let fours = [pos(0, 3), pos(0, 4)];
    let mut focus = fours.to_vec();
    focus.extend([(7, 3), (7, 4), (7, 8), (7, 9), (7, 12), (8, 12), (12, 12), (13, 12)].map(|(r, c)| pos(r, c)));
    let run = |moves_left_effect| {
        let config = MctsConfig { simulations: 4000, moves_left_effect, ..Default::default() };
        let mut rng = StdRng::seed_from_u64(0);
        Mcts::new(config).search(&b, &mut LosingWhiteEvaluator { focus: focus.clone() }, &mut rng)
    };
    let share = |result: &SearchResult| {
        let total: u32 = result.visits.iter().map(|&(_, n)| n).sum();
        let fours: u32 = result.visits.iter().filter(|(mv, _)| fours.contains(mv)).map(|&(_, n)| n).sum();
        fours as f32 / total as f32
    };

    let plain = run(0.0);
    let prolong = run(0.05);
    assert!(fours.contains(&prolong.best_move()), "best = {}", prolong.best_move());
    assert!(share(&prolong) > share(&plain) + 0.1, "share {} -> {}", share(&plain), share(&prolong));
}
