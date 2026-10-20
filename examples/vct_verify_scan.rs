//! vct_scan の --out の AllowedAvoidable を確かめ直す。
//! 1. 防げたとされた手 (safe) のあと、絞らない読みでも相手に VCT がないか
//! 2. AI の手 (played) のあとの相手の VCT が、受ける側の全ての合法手で崩れないか
use gomoku_engine::vct::{self, Limits, Outcome};
use gomoku_engine::{Board, GameResult, Stone, pos};

fn parse(text: &str, side: &str) -> Board {
    let (mut black, mut white) = (vec![], vec![]);
    for (r, line) in text.lines().skip(1).enumerate() {
        for (c, mark) in line[3..].chars().skip(1).step_by(3).enumerate() {
            match mark {
                'x' | 'X' => black.push(pos(r, c)),
                'o' | 'O' => white.push(pos(r, c)),
                _ => {}
            }
        }
    }
    Board::from_stones(&black, &white, if side == "black" { Stone::Black } else { Stone::White })
}

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let findings: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let wide = Limits { nodes: 1_000_000, focus: false, ..Limits::default() };
    for f in findings.iter().filter(|f| f["verdict"] == "allowed_avoidable") {
        let board = parse(f["board"].as_str().unwrap(), f["side"].as_str().unwrap());
        let at = |v: &serde_json::Value| pos(v[1].as_u64().unwrap() as usize, v[0].as_u64().unwrap() as usize);
        let (played, safe) = (at(&f["played"]), at(&f["safe"]));
        let mut after_safe = board.clone();
        after_safe.play(safe).unwrap();
        let safe_check = vct::solve(&after_safe, wide);
        // played のあと相手の VCT の最初の手 t を打ち、受ける側 (AI) の全ての手のあとも相手に VCT が残るか
        let mut after = board.clone();
        after.play(played).unwrap();
        let t = vct::find(&after, Limits::default()).unwrap();
        let mut b = after.clone();
        let opp = b.side_to_move();
        let mut holes = 0;
        if b.play(t).unwrap() == GameResult::Ongoing {
            for q in b.legal_moves() {
                let mut c = b.clone();
                match c.play(q).unwrap() {
                    GameResult::Ongoing => {
                        let wins = |l| matches!(vct::solve(&c, l), Outcome::Win(_));
                        if c.winning_moves(opp).is_empty() && !wins(Limits::default()) && !wins(wide) {
                            holes += 1;
                        }
                    }
                    GameResult::Win(w) if w != opp => holes += 1,
                    _ => {}
                }
            }
        }
        println!("{} played {:?} safe {:?}: safe 再確認 {:?} / 相手の VCT を崩せる手 {holes}",
            f["source"].as_str().unwrap(), f["played"], f["safe"], safe_check);
    }
}
