//! バッチ推論に対応した PUCT 探索 (AlphaZero 方式)。
//!
//! 葉ノードを最大 `batch_size` 個まとめて `Evaluator` に渡す。
//! 同じ葉に探索が集中しないよう、選択中の経路には仮想損失をかける。

use rand::Rng;
use rand_distr::{Distribution, Gamma};

use crate::board::{Board, CELLS, GameResult};

/// ネットワークによる局面評価の結果。
pub struct Evaluation {
    /// 各点の方策ロジット (長さ CELLS)。非合法手の値は無視される。
    pub policy: Vec<f32>,
    /// 手番側から見た価値 [-1, 1]
    pub value: f32,
}

pub trait Evaluator {
    fn evaluate(&mut self, boards: &[Board]) -> Vec<Evaluation>;
}

/// 全合法手を等確率・価値 0 とする評価器(テスト・ベースライン用)。
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate(&mut self, boards: &[Board]) -> Vec<Evaluation> {
        boards
            .iter()
            .map(|_| Evaluation { policy: vec![0.0; CELLS], value: 0.0 })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct MctsConfig {
    pub simulations: u32,
    pub batch_size: usize,
    pub c_puct: f32,
    /// 未訪問の子の Q を親の Q からどれだけ下げるか
    pub fpu_reduction: f32,
    pub dirichlet_alpha: f32,
    /// ルートの事前確率に混ぜるノイズの割合。0 で無効(対戦時)。
    pub dirichlet_epsilon: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            simulations: 800,
            batch_size: 16,
            c_puct: 1.5,
            fpu_reduction: 0.2,
            dirichlet_alpha: 0.03,
            dirichlet_epsilon: 0.0,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum NodeState {
    Unexpanded,
    /// 評価待ちとしてバッチに入っている
    Pending,
    Expanded,
    /// 終局。この手を打った側から見た価値を持つ。
    Terminal(f32),
}

struct Node {
    mv: usize,
    prior: f32,
    visits: u32,
    /// この手を打った側から見た価値の合計
    value_sum: f32,
    virtual_loss: u32,
    first_child: usize,
    num_children: usize,
    state: NodeState,
}

impl Node {
    fn new(mv: usize, prior: f32) -> Self {
        Self {
            mv,
            prior,
            visits: 0,
            value_sum: 0.0,
            virtual_loss: 0,
            first_child: 0,
            num_children: 0,
            state: NodeState::Unexpanded,
        }
    }
}

pub struct SearchResult {
    /// ルートの各合法手と訪問回数
    pub visits: Vec<(usize, u32)>,
    /// ルートの手番側から見た価値
    pub value: f32,
}

impl SearchResult {
    pub fn best_move(&self) -> usize {
        self.visits.iter().max_by_key(|&&(_, n)| n).expect("合法手がない").0
    }

    /// 訪問回数^(1/temperature) に比例して手を選ぶ。temperature <= 0 なら最多訪問手。
    pub fn sample_move<R: Rng>(&self, temperature: f32, rng: &mut R) -> usize {
        if temperature <= 0.0 {
            return self.best_move();
        }
        let weights: Vec<f64> =
            self.visits.iter().map(|&(_, n)| (n as f64).powf(1.0 / temperature as f64)).collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return self.best_move();
        }
        let mut x = rng.random::<f64>() * total;
        for (&(mv, _), w) in self.visits.iter().zip(&weights) {
            x -= w;
            if x <= 0.0 {
                return mv;
            }
        }
        self.visits.last().unwrap().0
    }

    /// 訪問回数を正規化した方策 (長さ CELLS)。学習の教師データになる。
    pub fn policy(&self) -> Vec<f32> {
        let total: u32 = self.visits.iter().map(|&(_, n)| n).sum();
        let mut policy = vec![0.0; CELLS];
        for &(mv, n) in &self.visits {
            policy[mv] = n as f32 / total.max(1) as f32;
        }
        policy
    }
}

/// 探索する手の候補。即勝ちがあればその 1 手、相手の五を止める必要があれば止める手だけに絞る。
fn candidate_moves(board: &Board) -> Vec<usize> {
    let side = board.side_to_move();
    if board.move_count() > 0 {
        if let Some(&win) = board.winning_moves(side).first() {
            return vec![win];
        }
        let blocks: Vec<usize> =
            board.winning_moves(side.opponent()).into_iter().filter(|&p| board.is_legal(p)).collect();
        // 止める点が禁じ手しかなければ負けは確定なので、通常の合法手から選ばせる
        if !blocks.is_empty() {
            return blocks;
        }
    }
    board.legal_moves()
}

pub struct Mcts {
    config: MctsConfig,
    nodes: Vec<Node>,
}

const ROOT: usize = 0;

impl Mcts {
    pub fn new(config: MctsConfig) -> Self {
        Self { config, nodes: Vec::new() }
    }

    pub fn config(&self) -> &MctsConfig {
        &self.config
    }

    pub fn search<E: Evaluator + ?Sized, R: Rng>(
        &mut self,
        root: &Board,
        evaluator: &mut E,
        rng: &mut R,
    ) -> SearchResult {
        assert_eq!(root.result(), GameResult::Ongoing, "終局した局面は探索できない");
        self.nodes.clear();
        self.nodes.push(Node::new(usize::MAX, 1.0));

        let eval = evaluator.evaluate(std::slice::from_ref(root)).pop().expect("評価結果がない");
        self.expand(ROOT, root, &eval.policy);
        if self.config.dirichlet_epsilon > 0.0 {
            self.add_root_noise(rng);
        }

        let mut done = 0;
        while done < self.config.simulations {
            let want = self.config.batch_size.min((self.config.simulations - done) as usize);
            let mut paths = Vec::with_capacity(want);
            let mut boards = Vec::with_capacity(want);
            for _ in 0..want {
                let (path, board) = self.select(root);
                let leaf = *path.last().unwrap();
                match self.nodes[leaf].state {
                    NodeState::Terminal(v) => {
                        self.backup(&path, v);
                        done += 1;
                    }
                    NodeState::Pending => {
                        // 既にバッチ内にある葉を選んだら、このバッチはここで打ち切る
                        self.revert(&path);
                        break;
                    }
                    NodeState::Unexpanded => {
                        self.nodes[leaf].state = NodeState::Pending;
                        paths.push(path);
                        boards.push(board);
                    }
                    NodeState::Expanded => unreachable!("select は展開済みノードで止まらない"),
                }
            }
            if boards.is_empty() {
                continue;
            }
            let evals = evaluator.evaluate(&boards);
            for ((path, board), eval) in paths.iter().zip(&boards).zip(&evals) {
                self.expand(*path.last().unwrap(), board, &eval.policy);
                // eval.value は葉の手番側から見た値なので、葉に打った側からは反転する
                self.backup(path, -eval.value);
                done += 1;
            }
        }

        let root_node = &self.nodes[ROOT];
        SearchResult {
            visits: self.children(ROOT).map(|c| (self.nodes[c].mv, self.nodes[c].visits)).collect(),
            value: -root_node.value_sum / root_node.visits.max(1) as f32,
        }
    }

    fn children(&self, id: usize) -> std::ops::Range<usize> {
        let n = &self.nodes[id];
        n.first_child..n.first_child + n.num_children
    }

    /// ルートから葉まで降り、経路とその葉の局面を返す。経路上には仮想損失を加える。
    fn select(&mut self, root: &Board) -> (Vec<usize>, Board) {
        let mut board = root.clone();
        let mut path = vec![ROOT];
        let mut id = ROOT;
        loop {
            self.nodes[id].virtual_loss += 1;
            if self.nodes[id].state != NodeState::Expanded {
                return (path, board);
            }
            id = self.best_child(id);
            board.play(self.nodes[id].mv).expect("子ノードは合法手のみ");
            path.push(id);
            if self.nodes[id].state == NodeState::Unexpanded {
                match board.result() {
                    GameResult::Ongoing => {}
                    GameResult::Win(_) => self.nodes[id].state = NodeState::Terminal(1.0),
                    GameResult::Draw => self.nodes[id].state = NodeState::Terminal(0.0),
                }
            }
        }
    }

    fn best_child(&self, id: usize) -> usize {
        let parent = &self.nodes[id];
        let parent_n = parent.visits + parent.virtual_loss;
        let sqrt_n = (parent_n as f32).sqrt();
        // 親の value_sum は親に打った側の視点なので、子を選ぶ側の視点では反転する
        let parent_q = if parent.visits > 0 { -parent.value_sum / parent.visits as f32 } else { 0.0 };
        let fpu = parent_q - self.config.fpu_reduction;

        let score = |c: usize| {
            let child = &self.nodes[c];
            let n = child.visits + child.virtual_loss;
            let q = if n == 0 {
                fpu
            } else {
                // 仮想損失は訪問 1 回・価値 -1 として扱う
                (child.value_sum - child.virtual_loss as f32) / n as f32
            };
            q + self.config.c_puct * child.prior * sqrt_n / (1 + n) as f32
        };
        self.children(id)
            .max_by(|&a, &b| score(a).total_cmp(&score(b)))
            .expect("展開済みノードに子がない")
    }

    fn expand(&mut self, id: usize, board: &Board, logits: &[f32]) {
        let moves = candidate_moves(board);
        let max = moves.iter().map(|&m| logits[m]).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = moves.iter().map(|&m| (logits[m] - max).exp()).collect();
        let total: f32 = exps.iter().sum();

        let first_child = self.nodes.len();
        self.nodes.extend(moves.iter().zip(&exps).map(|(&m, &e)| Node::new(m, e / total)));
        let node = &mut self.nodes[id];
        node.first_child = first_child;
        node.num_children = moves.len();
        node.state = NodeState::Expanded;
    }

    fn add_root_noise<R: Rng>(&mut self, rng: &mut R) {
        let gamma = Gamma::new(self.config.dirichlet_alpha, 1.0).expect("dirichlet_alpha は正の値");
        let noise: Vec<f32> = self.children(ROOT).map(|_| gamma.sample(rng)).collect();
        let total: f32 = noise.iter().sum();
        if total <= 0.0 {
            return;
        }
        let eps = self.config.dirichlet_epsilon;
        for (c, n) in self.children(ROOT).zip(noise) {
            let prior = &mut self.nodes[c].prior;
            *prior = (1.0 - eps) * *prior + eps * n / total;
        }
    }

    /// 葉から根へ価値を伝播し、仮想損失を取り除く。value は葉に打った側から見た値。
    fn backup(&mut self, path: &[usize], mut value: f32) {
        for &id in path.iter().rev() {
            let node = &mut self.nodes[id];
            node.visits += 1;
            node.value_sum += value;
            node.virtual_loss -= 1;
            value = -value;
        }
    }

    fn revert(&mut self, path: &[usize]) {
        for &id in path {
            self.nodes[id].virtual_loss -= 1;
        }
    }
}
