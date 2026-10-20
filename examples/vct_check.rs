//! VCT ありと判断した局面で、受ける側の全ての合法手を試し、どの手のあとも攻める側に VCT が残るかを確かめる
//! (受けの候補の漏れを探す)。使い方: vct_check <record.json>...
use gomoku_engine::vct::{self, Limits, Outcome};
use gomoku_engine::{Board, GameResult, pos};

/// p を打ったあと、受ける側のどの手にも攻める側が勝てるか。崩せた手を返す。
fn refutations(board: &Board, p: usize) -> Vec<usize> {
    let me = board.side_to_move();
    let mut after = board.clone();
    if after.play(p).unwrap() != GameResult::Ongoing {
        return vec![];
    }
    let mut found = vec![];
    for q in after.legal_moves() {
        let mut b = after.clone();
        match b.play(q).unwrap() {
            GameResult::Win(w) if w != me => found.push(q),
            GameResult::Ongoing => {
                // 攻める側の五、または VCT が残っているか
                // 2 手目以降の攻めを絞った読みで見つからなければ、絞らずに読み直す
                let wins = |limits| matches!(vct::solve(&b, limits), Outcome::Win(_));
                if b.winning_moves(me).is_empty()
                    && !wins(Limits::default())
                    && !wins(Limits { nodes: 2_000_000, focus: false, ..Limits::default() })
                {
                    found.push(q);
                }
            }
            _ => {}
        }
    }
    found
}

fn main() {
    let (mut checked, mut bad) = (0, 0);
    for path in std::env::args().skip(1) {
        let rec: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let mut board = Board::new();
        for m in rec["moves"].as_array().unwrap() {
            if board.move_count() > 0 {
                if let Outcome::Win(p) = vct::solve(&board, Limits::default()) {
                    checked += 1;
                    let refs = refutations(&board, p);
                    if !refs.is_empty() {
                        bad += 1;
                        let xy = |p: usize| (p % 15, p / 15);
                        println!("{path} {} 手目 ({:?}): {:?} を {:?} で崩せる", board.move_count() + 1,
                            board.side_to_move(), xy(p), refs.iter().map(|&q| xy(q)).collect::<Vec<_>>());
                    }
                }
            }
            let p = pos(m["y"].as_u64().unwrap() as usize, m["x"].as_u64().unwrap() as usize);
            if board.play(p).is_err() || board.result() != GameResult::Ongoing {
                break;
            }
        }
    }
    println!("VCT ありの局面 {checked} / 崩せたもの {bad}");
}
