use rand::SeedableRng;
use rand::rngs::StdRng;

use super::board;
use super::vcf::after;
use crate::features::{BASE_PLANES, FEATURE_LEN, VCF_BUDGET, encode};
use crate::vcf;
use crate::npz::{NpyArray, NpyData, write_npz};
use crate::selfplay::{OpeningBalance, PlayoutCap, balanced_opening, play_game, random_opening};
use crate::{Board, CELLS, CENTER, GameResult, Mcts, MctsConfig, Stone, UniformEvaluator, pos};

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
fn vcf_planes_mark_the_attacking_moves_of_each_side() {
    // VCF の攻め手 (手順の偶数番目) の点だけが 1 の平面
    let attacks = |b: &Board| {
        let mut plane = vec![0u8; CELLS];
        for p in vcf::find(b, VCF_BUDGET).unwrap_or_default().into_iter().step_by(2) {
            plane[p] = 1;
        }
        plane
    };
    // 41 手目 (黒番): 黒に VCF がある
    let board = after(40);
    let mut passed = board.clone();
    passed.pass();
    let black_vcf = attacks(&board);
    assert!(black_vcf.contains(&1));

    let mut f = vec![0; FEATURE_LEN];
    encode(&board, &mut f);
    assert_eq!(plane(&f, 4), black_vcf);
    assert_eq!(plane(&f, 5), attacks(&passed));

    // 黒が一手休んだ局面 (白番) では、黒の VCF は相手の平面に出る
    encode(&passed, &mut f);
    assert_eq!(plane(&f, 4), attacks(&passed));
    assert_eq!(plane(&f, 5), black_vcf);
}

#[test]
fn base_planes_are_the_first_planes_of_the_full_features() {
    let board = after(40);
    let mut full = vec![0; FEATURE_LEN];
    let mut base = vec![0; BASE_PLANES * CELLS];
    encode(&board, &mut full);
    encode(&board, &mut base);
    assert_eq!(base, full[..BASE_PLANES * CELLS]);
}

#[test]
fn self_play_game_labels_values_from_side_to_move() {
    let mut rng = StdRng::seed_from_u64(0);
    let mut mcts = Mcts::new(MctsConfig { simulations: 32, dirichlet_epsilon: 0.25, ..Default::default() });
    let game = play_game(&mut mcts, &mut UniformEvaluator, &Board::new(), 10, PlayoutCap::OFF, &mut rng);

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
        &Board::new(),
        4,
        &mut rng,
    );
    assert_ne!(result, GameResult::Ongoing);
    assert!(black_calls.get() > 0 && white_calls.get() > 0);
}

#[test]
fn random_opening_plays_up_to_max_moves_near_the_stones() {
    let mut rng = StdRng::seed_from_u64(0);
    assert_eq!(random_opening(0, &mut rng).move_count(), 0);

    let mut seen = [false; 5];
    for _ in 0..200 {
        let b = random_opening(4, &mut rng);
        assert_eq!(b.result(), GameResult::Ongoing);
        // 0 手なら初期局面、それ以外は天元 + ランダムな手
        let random_moves = b.move_count().saturating_sub(1);
        assert!(random_moves <= 4 && b.move_count() != 1);
        seen[random_moves] = true;
        if b.move_count() > 0 {
            assert_eq!(b.get(CENTER), Some(Stone::Black));
            let expected = if b.move_count() % 2 == 0 { Stone::Black } else { Stone::White };
            assert_eq!(b.side_to_move(), expected);
        }
        // どの石も他の石から 2 路以内にある
        let stones: Vec<usize> = (0..CELLS).filter(|&p| b.get(p).is_some()).collect();
        for &p in &stones {
            let near = |&q: &usize| q != p && (p / 15).abs_diff(q / 15) <= 2 && (p % 15).abs_diff(q % 15) <= 2;
            assert!(stones.len() < 2 || stones.iter().any(near));
        }
    }
    assert!(seen.iter().all(|&s| s));
}

#[test]
fn balanced_opening_keeps_only_openings_searched_as_even() {
    // 石が limit 個以下の局面は互角 (価値 0)、それより多い局面は一方的 (0.9) と評価する
    struct ByStones(usize);
    impl crate::Evaluator for ByStones {
        fn evaluate(&mut self, boards: &[Board]) -> Vec<crate::Evaluation> {
            let value = |b: &Board| if b.move_count() <= self.0 { 0.0 } else { 0.9 };
            boards.iter().map(|b| crate::Evaluation { policy: vec![0.0; CELLS], value: value(b) }).collect()
        }
    }
    let mut rng = StdRng::seed_from_u64(2);
    // 1 回の探索では葉 (開局 + 1 手) を 1 つだけ評価するので、開局の石が 2 個以下なら互角になる
    let mut mcts = Mcts::new(MctsConfig { simulations: 1, ..Default::default() });
    let balance = OpeningBalance { max_value: 0.5, simulations: 1, games: 1.0 };
    for _ in 0..50 {
        assert!(balanced_opening(4, balance, &mut mcts, &mut ByStones(3), &mut rng).move_count() <= 2);
    }
    // 無効なら評価せずにランダムな開局をそのまま使う
    let counts: Vec<usize> = (0..50)
        .map(|_| balanced_opening(4, OpeningBalance::OFF, &mut mcts, &mut ByStones(3), &mut rng).move_count())
        .collect();
    assert!(counts.iter().any(|&n| n > 2));
    // games の割合の局だけ互角の開局にする
    let half = OpeningBalance { games: 0.5, ..balance };
    let uneven = (0..200)
        .filter(|_| balanced_opening(4, half, &mut mcts, &mut ByStones(3), &mut rng).move_count() > 2)
        .count();
    // 互角でない開局 (石が 3 個以上) は random_opening の約 6 割なので、半分の局では約 60 局
    assert!((30..=90).contains(&uneven), "{uneven}");
    // 互角の開局が見つからなくても、試した中から返す
    assert_eq!(balanced_opening(4, balance, &mut mcts, &mut ByStones(0), &mut rng).result(), GameResult::Ongoing);
}

#[test]
fn self_play_from_an_opening_records_only_searched_positions() {
    let mut rng = StdRng::seed_from_u64(1);
    let mut mcts = Mcts::new(MctsConfig { simulations: 32, ..Default::default() });
    let opening = board(&[(7, 7), (8, 8)], &[(6, 7)], Stone::White);
    let game = play_game(&mut mcts, &mut UniformEvaluator, &opening, 4, PlayoutCap::OFF, &mut rng);

    assert_eq!(game.samples.len(), game.moves - 3);
    // 最初の記録は開局直後の白番の局面
    let first = &game.samples[0].features;
    assert!(plane(first, 2).iter().all(|&x| x == 0));
    assert_eq!(plane(first, 0)[pos(6, 7)], 1);
    assert_eq!(plane(first, 1)[pos(7, 7)], 1);
}

#[test]
fn playout_cap_records_only_full_searches() {
    use std::cell::Cell;
    struct Counting<'a>(&'a Cell<usize>);
    impl crate::Evaluator for Counting<'_> {
        fn evaluate(&mut self, boards: &[crate::Board]) -> Vec<crate::Evaluation> {
            self.0.set(self.0.get() + boards.len());
            UniformEvaluator.evaluate(boards)
        }
    }
    let mut rng = StdRng::seed_from_u64(0);
    let mut mcts = Mcts::new(MctsConfig { simulations: 64, batch_size: 1, dirichlet_epsilon: 0.25, ..Default::default() });

    // 速い探索だけなら学習データは残らず、評価回数は 1 手あたり (ルート 1 回 + 探索回数) 以下になる
    let calls = Cell::new(0);
    let cap = PlayoutCap { full_prob: 0.0, fast_simulations: 4 };
    let game = play_game(&mut mcts, &mut Counting(&calls), &Board::new(), 4, cap, &mut rng);
    assert_ne!(game.result, GameResult::Ongoing);
    assert!(game.samples.is_empty());
    assert!(calls.get() <= game.moves * 5);

    // 半分ほどを通常の探索にすると、その局面だけが記録される
    let cap = PlayoutCap { full_prob: 0.5, fast_simulations: 4 };
    let mut recorded = 0;
    let mut moves = 0;
    for _ in 0..10 {
        let game = play_game(&mut mcts, &mut UniformEvaluator, &Board::new(), 4, cap, &mut rng);
        recorded += game.samples.len();
        moves += game.moves - 1;
    }
    assert!(recorded > 0 && recorded < moves);
}
