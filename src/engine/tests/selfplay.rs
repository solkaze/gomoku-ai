use rand::SeedableRng;
use rand::rngs::StdRng;

use super::board;
use crate::features::{FEATURE_LEN, encode};
use crate::npz::{NpyArray, NpyData, write_npz};
use crate::selfplay::play_game;
use crate::{CELLS, GameResult, Mcts, MctsConfig, Stone, UniformEvaluator, pos};

fn plane(features: &[u8], i: usize) -> &[u8] {
    &features[i * CELLS..(i + 1) * CELLS]
}

#[test]
fn features_are_relative_to_side_to_move() {
    // 黒番: (7,7) は三三の禁点
    let b = board(&[(7, 5), (7, 6), (5, 7), (6, 7)], &[(0, 0)], Stone::Black);
    let mut f = vec![0; FEATURE_LEN];
    encode(&b, &mut f);
    assert_eq!(plane(&f, 0)[pos(7, 5)], 1);
    assert_eq!(plane(&f, 1)[pos(0, 0)], 1);
    assert!(plane(&f, 2).iter().all(|&x| x == 1));
    assert_eq!(plane(&f, 3).iter().map(|&x| x as usize).sum::<usize>(), 1);
    assert_eq!(plane(&f, 3)[pos(7, 7)], 1);

    // 白番: 自分と相手が入れ替わり、禁点は黒のものを示し続ける
    let w = board(&[(7, 5), (7, 6), (5, 7), (6, 7)], &[(0, 0)], Stone::White);
    encode(&w, &mut f);
    assert_eq!(plane(&f, 0)[pos(0, 0)], 1);
    assert_eq!(plane(&f, 1)[pos(7, 5)], 1);
    assert!(plane(&f, 2).iter().all(|&x| x == 0));
    assert_eq!(plane(&f, 3)[pos(7, 7)], 1);
}

#[test]
fn self_play_game_labels_values_from_side_to_move() {
    let mut rng = StdRng::seed_from_u64(0);
    let mut mcts = Mcts::new(MctsConfig { simulations: 32, dirichlet_epsilon: 0.25, ..Default::default() });
    let game = play_game(&mut mcts, &mut UniformEvaluator, 10, &mut rng);

    assert_ne!(game.result, GameResult::Ongoing);
    // 初手は記録しない
    assert_eq!(game.samples.len(), game.moves - 1);
    for (i, sample) in game.samples.iter().enumerate() {
        assert_eq!(sample.features.len(), FEATURE_LEN);
        assert!((sample.policy.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        // samples[i] は (i + 1) 手目、つまり i が偶数なら白番
        let side = if i % 2 == 0 { Stone::White } else { Stone::Black };
        let expected = match game.result {
            GameResult::Win(winner) if winner == side => 1.0,
            GameResult::Win(_) => -1.0,
            _ => 0.0,
        };
        assert_eq!(sample.value, expected);
    }
}

#[test]
fn npz_header_is_aligned() {
    let dir = std::env::temp_dir().join(format!("gomoku-npz-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("t.npz");
    let data = [1.0f32, 2.0, 3.0];
    write_npz(&path, &[NpyArray { name: "value", shape: vec![3], data: NpyData::F32(&data) }]).unwrap();

    let mut zip = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut zip.by_name("value.npy").unwrap(), &mut bytes).unwrap();
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    assert_eq!((10 + header_len) % 64, 0);
    assert!(std::str::from_utf8(&bytes[10..10 + header_len]).unwrap().contains("'shape': (3,)"));
    assert_eq!(bytes.len(), 10 + header_len + 12);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn match_uses_each_sides_evaluator() {
    use std::cell::Cell;
    struct Counting<'a>(&'a Cell<usize>);
    impl crate::Evaluator for Counting<'_> {
        fn evaluate(&mut self, boards: &[crate::Board]) -> Vec<crate::Evaluation> {
            self.0.set(self.0.get() + boards.len());
            UniformEvaluator.evaluate(boards)
        }
    }
    let (black_calls, white_calls) = (Cell::new(0), Cell::new(0));
    let mut rng = StdRng::seed_from_u64(0);
    let mut mcts = Mcts::new(MctsConfig { simulations: 16, ..Default::default() });
    let result = crate::selfplay::play_match(
        &mut mcts,
        &mut Counting(&black_calls),
        &mut Counting(&white_calls),
        4,
        &mut rng,
    );
    assert_ne!(result, GameResult::Ongoing);
    assert!(black_calls.get() > 0 && white_calls.get() > 0);
}
