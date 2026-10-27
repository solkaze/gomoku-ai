//! バッチ推論に対応した PUCT 探索 (AlphaZero 方式)。
//!
//! 葉ノードを最大 `batch_size` 個まとめて `Evaluator` に渡す。
//! 同じ葉に探索が集中しないよう、選択中の経路には仮想損失をかける。

use std::sync::atomic::{AtomicBool, Ordering};

use rand::Rng;
use rand_distr::{Distribution, Gamma};

use crate::board::{Board, CELLS, GameResult, Stone};
use crate::vcf;

/// ネットワークによる局面評価の結果。
pub struct Evaluation {
    /// 各点の方策ロジット (長さ CELLS)。探索の候補にならない手の値は無視される。
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
    /// ルートで VCF (四の連続での勝ち) を読む。自分に VCF があればその手だけを、なければ相手に VCF を許さない
    /// 手だけを候補にする (全ての手が許すなら絞らない)。
    pub vcf: bool,
    /// 勝ち負けがはっきりした局面 (|Q| が MOVES_LEFT_THRESHOLD を超える) で、負けているときは終局を遠ざける手、
    /// 勝っているときは早く終わる手を選ぶ強さ。残り手数の差に応じて Q に足す値の上限。0 で無効。
    /// 負けの局面では評価がどの手も -1 近くに張り付き、粘る手を選ぶ理由がなくなるため (対戦用)。
    pub moves_left_effect: f32,
}

/// 残り手数による補正をかけ始める |Q|。これを超えた分に比例して強くし、|Q| = 1 で moves_left_effect まで
const MOVES_LEFT_THRESHOLD: f32 = 0.8;
/// 残り手数 1 手の差あたりの補正
const MOVES_LEFT_SLOPE: f32 = 0.005;
/// ネットワークで評価した葉から終局までの手数の見積もり (終局の葉は 0)。
/// 読み切れていない手順を、読み切った手順より長いものとして扱う
const UNKNOWN_MOVES_LEFT: f32 = 10.0;

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            simulations: 800,
            batch_size: 16,
            c_puct: 1.5,
            fpu_reduction: 0.2,
            dirichlet_alpha: 0.03,
            dirichlet_epsilon: 0.0,
            vcf: false,
            moves_left_effect: 0.0,
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
    /// この手の後の局面から終局までの手数の見積もりの合計
    moves_left_sum: f32,
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
            moves_left_sum: 0.0,
            virtual_loss: 0,
            first_child: 0,
            num_children: 0,
            state: NodeState::Unexpanded,
        }
    }
}

pub struct SearchResult {
    /// ルートの各候補手と訪問回数
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

/// ルートの候補手を VCF の読みで絞る。自分に VCF があればその最初の手だけ、なければ相手に VCF を許さない手だけ。
fn vcf_root_moves(board: &Board, moves: Vec<usize>) -> Vec<usize> {
    if moves.len() <= 1 {
        return moves;
    }
    if let Some(line) = vcf::find(board, vcf::DEFAULT_BUDGET) {
        return vec![line[0]];
    }
    let safe = vcf::safe_moves(board, &moves, vcf::DEFAULT_BUDGET);
    if safe.is_empty() { moves } else { safe }
}

/// 探索する手の候補。即勝ちがあればその 1 手、相手の五を止める必要があれば止める手だけに絞る。
/// 黒の禁じ手は打てば即負けなので、他に打てる点がある限り候補にしない。
fn candidate_moves(board: &Board) -> Vec<usize> {
    let side = board.side_to_move();
    let safe = |&p: &usize| side != Stone::Black || !board.is_forbidden(p);
    if board.move_count() > 0 {
        if let Some(&win) = board.winning_moves(side).first() {
            return vec![win];
        }
        let blocks: Vec<usize> = board.winning_moves(side.opponent()).into_iter().filter(safe).collect();
        // 止める点が禁じ手しかなければ負けは確定なので、禁じ手以外の手から選ばせる
        if !blocks.is_empty() {
            return blocks;
        }
    }
    let legal = board.legal_moves();
    let moves: Vec<usize> = legal.iter().copied().filter(safe).collect();
    // 禁じ手しか残っていなければ、それを打って負けるしかない
    if moves.is_empty() { legal } else { moves }
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
        self.search_until(root, evaluator, rng, &AtomicBool::new(false))
    }

    /// search と同じだが、stop が立ったらそこまでの結果で打ち切る。
    pub fn search_until<E: Evaluator + ?Sized, R: Rng>(
        &mut self,
        root: &Board,
        evaluator: &mut E,
        rng: &mut R,
        stop: &AtomicBool,
    ) -> SearchResult {
        let (simulations, root_noise) = (self.config.simulations, self.config.dirichlet_epsilon > 0.0);
        self.run(root, evaluator, rng, stop, simulations, root_noise)
    }

    /// 探索回数とルートノイズの有無だけを変えて探索する (playout cap randomization の速い探索用)。
    pub fn search_with<E: Evaluator + ?Sized, R: Rng>(
        &mut self,
        root: &Board,
        evaluator: &mut E,
        rng: &mut R,
        simulations: u32,
        root_noise: bool,
    ) -> SearchResult {
        let root_noise = root_noise && self.config.dirichlet_epsilon > 0.0;
        self.run(root, evaluator, rng, &AtomicBool::new(false), simulations, root_noise)
    }

    fn run<E: Evaluator + ?Sized, R: Rng>(
        &mut self,
        root: &Board,
        evaluator: &mut E,
        rng: &mut R,
        stop: &AtomicBool,
        simulations: u32,
        root_noise: bool,
    ) -> SearchResult {
        assert_eq!(root.result(), GameResult::Ongoing, "終局した局面は探索できない");
        self.nodes.clear();
        self.nodes.push(Node::new(usize::MAX, 1.0));

        let eval = evaluator.evaluate(std::slice::from_ref(root)).pop().expect("評価結果がない");
        let mut moves = candidate_moves(root);
        if self.config.vcf {
            moves = vcf_root_moves(root, moves);
        }
        self.expand_moves(ROOT, &moves, &eval.policy);
        if root_noise {
            self.add_root_noise(rng);
        }

        let mut done = 0;
        while done < simulations && !stop.load(Ordering::Relaxed) {
            let want = self.config.batch_size.min((simulations - done) as usize);
            let mut paths = Vec::with_capacity(want);
            let mut boards = Vec::with_capacity(want);
            for _ in 0..want {
                let (path, board) = self.select(root);
                let leaf = *path.last().unwrap();
                match self.nodes[leaf].state {
                    NodeState::Terminal(v) => {
                        self.backup(&path, v, 0.0);
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
                self.backup(path, -eval.value, UNKNOWN_MOVES_LEFT);
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
            let mover = board.side_to_move();
            board.play(self.nodes[id].mv).expect("子ノードは合法手のみ");
            path.push(id);
            if self.nodes[id].state == NodeState::Unexpanded {
                match board.result() {
                    GameResult::Ongoing => {}
                    // 禁じ手を打った場合は、打った側の負けになる
                    GameResult::Win(winner) => {
                        self.nodes[id].state = NodeState::Terminal(if winner == mover { 1.0 } else { -1.0 })
                    }
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
        let moves_left = self.moves_left_bonus(parent, parent_q);

        let score = |c: usize| {
            let child = &self.nodes[c];
            let n = child.visits + child.virtual_loss;
            let q = if n == 0 {
                fpu
            } else {
                // 仮想損失は訪問 1 回・価値 -1 として扱う
                (child.value_sum - child.virtual_loss as f32) / n as f32
            };
            q + moves_left(child) + self.config.c_puct * child.prior * sqrt_n / (1 + n) as f32
        };
        self.children(id)
            .max_by(|&a, &b| score(a).total_cmp(&score(b)))
            .expect("展開済みノードに子がない")
    }

    /// 子を選ぶ側から見た Q が parent_q のとき、子の残り手数に応じて Q に足す値。負けていれば親の見積もりより
    /// 長い子、勝っていれば短い子ほど大きい。未訪問の子は 0。
    fn moves_left_bonus(&self, parent: &Node, parent_q: f32) -> impl Fn(&Node) -> f32 {
        let effect = self.config.moves_left_effect;
        let strength = if effect > 0.0 && parent.visits > 0 {
            ((parent_q.abs() - MOVES_LEFT_THRESHOLD) / (1.0 - MOVES_LEFT_THRESHOLD)).max(0.0)
        } else {
            0.0
        };
        // 親の局面から終局までの手数から、子の手の 1 手を引いたもの
        let expected = parent.moves_left_sum / parent.visits.max(1) as f32 - 1.0;
        let sign = if parent_q < 0.0 { 1.0 } else { -1.0 };
        move |child: &Node| {
            if strength == 0.0 || child.visits == 0 {
                return 0.0;
            }
            let diff = child.moves_left_sum / child.visits as f32 - expected;
            sign * strength * (MOVES_LEFT_SLOPE * diff).clamp(-effect, effect)
        }
    }

    fn expand(&mut self, id: usize, board: &Board, logits: &[f32]) {
        self.expand_moves(id, &candidate_moves(board), logits);
    }

    fn expand_moves(&mut self, id: usize, moves: &[usize], logits: &[f32]) {
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

    /// 葉から根へ価値を伝播し、仮想損失を取り除く。value は葉に打った側から見た値、moves_left は葉の局面から
    /// 終局までの手数の見積もり。
    fn backup(&mut self, path: &[usize], mut value: f32, mut moves_left: f32) {
        for &id in path.iter().rev() {
            let node = &mut self.nodes[id];
            node.visits += 1;
            node.value_sum += value;
            node.moves_left_sum += moves_left;
            node.virtual_loss -= 1;
            value = -value;
            moves_left += 1.0;
        }
    }

    fn revert(&mut self, path: &[usize]) {
        for &id in path {
            self.nodes[id].virtual_loss -= 1;
        }
    }
}
