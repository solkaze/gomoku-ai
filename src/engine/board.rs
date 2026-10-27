use std::fmt;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::rules::{self, BLACK, EMPTY, Rules, WHITE};

pub const SIZE: usize = 15;
pub const CELLS: usize = SIZE * SIZE;
/// 黒の初手は必ず天元 (7, 7)
pub const CENTER: usize = pos(7, 7);

pub const fn pos(row: usize, col: usize) -> usize {
    row * SIZE + col
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stone {
    Black,
    White,
}

impl Stone {
    pub fn opponent(self) -> Self {
        match self {
            Stone::Black => Stone::White,
            Stone::White => Stone::Black,
        }
    }

    fn cell(self) -> u8 {
        match self {
            Stone::Black => BLACK,
            Stone::White => WHITE,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameResult {
    Ongoing,
    Win(Stone),
    /// 盤が埋まった
    Draw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveError {
    GameOver,
    OutOfBoard,
    Occupied,
    NotCenter,
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            MoveError::GameOver => "対局は終了しています",
            MoveError::OutOfBoard => "盤外です",
            MoveError::Occupied => "既に石があります",
            MoveError::NotCenter => "黒の初手は天元 (7, 7) です",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for MoveError {}

#[derive(Clone, Debug)]
pub struct Board {
    cells: [u8; CELLS],
    side: Stone,
    moves: usize,
    last_move: Option<usize>,
    result: GameResult,
    rules: Rules,
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    /// 全ての禁じ手を有効にした盤。
    pub fn new() -> Self {
        Self::with_rules(Rules::default())
    }

    /// 黒に適用する禁じ手を指定した盤。
    pub fn with_rules(rules: Rules) -> Self {
        Self {
            cells: [EMPTY; CELLS],
            side: Stone::Black,
            moves: 0,
            last_move: None,
            result: GameResult::Ongoing,
            rules,
        }
    }

    pub fn rules(&self) -> Rules {
        self.rules
    }

    /// 適用する禁じ手を変える。それまでの手の勝敗は判定し直さない。
    pub fn set_rules(&mut self, rules: Rules) {
        self.rules = rules;
    }

    /// 任意の局面を作る(検討・テスト用)。ルールや勝敗の検証はしない。
    pub fn from_stones(black: &[usize], white: &[usize], side: Stone) -> Self {
        let mut board = Self::new();
        for &p in black {
            board.cells[p] = BLACK;
        }
        for &p in white {
            board.cells[p] = WHITE;
        }
        board.side = side;
        board.moves = black.len() + white.len();
        board
    }

    pub fn get(&self, p: usize) -> Option<Stone> {
        match self.cells[p] {
            BLACK => Some(Stone::Black),
            WHITE => Some(Stone::White),
            _ => None,
        }
    }

    pub fn side_to_move(&self) -> Stone {
        self.side
    }

    pub fn move_count(&self) -> usize {
        self.moves
    }

    pub fn last_move(&self) -> Option<usize> {
        self.last_move
    }

    pub fn result(&self) -> GameResult {
        self.result
    }

    /// 黒が禁じ手を打って負けたか。
    pub fn lost_by_forbidden(&self) -> bool {
        self.result == GameResult::Win(Stone::White) && self.last_move.is_some_and(|p| self.cells[p] == BLACK)
    }

    /// 空点 p が黒にとって禁じ手か(手番に関係なく判定する)。
    pub fn is_forbidden(&self, p: usize) -> bool {
        let mut cells = self.cells;
        cells[p] == EMPTY && rules::is_forbidden(&mut cells, p, self.rules)
    }

    /// 黒の禁じ手の点をまとめて求める(手番に関係なく判定する)。
    pub fn forbidden_mask(&self) -> [bool; CELLS] {
        let mut cells = self.cells;
        std::array::from_fn(|p| cells[p] == EMPTY && rules::is_forbidden(&mut cells, p, self.rules))
    }

    /// stone が打てば五になる空点(黒の五は禁じ手より優先されるので常に合法)。
    /// 長連が禁じ手でなければ、黒の六以上の連も含む。
    pub fn winning_moves(&self, stone: Stone) -> Vec<usize> {
        let color = stone.cell();
        let mut cells = self.cells;
        (0..CELLS)
            .filter(|&p| {
                if cells[p] != EMPTY {
                    return false;
                }
                cells[p] = color;
                let five = rules::is_five(&cells, p, color, self.rules);
                cells[p] = EMPTY;
                five
            })
            .collect()
    }

    /// stone が打てば五になる空点のうち、p を通る 4 方向の線上 (p から 4 路以内) にあるもの。
    /// p に打った直後に、その手でできた四を調べるのに使う (winning_moves より速い)。
    pub fn winning_moves_through(&self, stone: Stone, p: usize) -> Vec<usize> {
        let color = stone.cell();
        let mut cells = self.cells;
        let (row, col) = ((p / SIZE) as isize, (p % SIZE) as isize);
        let mut found = Vec::new();
        for (dr, dc) in [(0, 1), (1, 0), (1, 1), (1, -1)] {
            for k in -4..=4 {
                let (r, c) = (row + dr * k, col + dc * k);
                if k == 0 || !(0..SIZE as isize).contains(&r) || !(0..SIZE as isize).contains(&c) {
                    continue;
                }
                let q = pos(r as usize, c as usize);
                if cells[q] != EMPTY || found.contains(&q) {
                    continue;
                }
                cells[q] = color;
                if rules::is_five(&cells, q, color, self.rules) {
                    found.push(q);
                }
                cells[q] = EMPTY;
            }
        }
        found
    }

    /// stone が打つと四 (あと 1 手で五) になりうる空点。連続する 5 点に stone の石が 3 つ・空点が 2 つあり、
    /// 相手の石がない並びの空点を集める。長連になるなどで実際には四にならない点も含むので、打って確かめること。
    pub fn four_candidates(&self, stone: Stone) -> Vec<usize> {
        self.window_candidates(stone, 3)
    }

    /// stone が打つと三になりうる空点。four_candidates と同じく、連続する 5 点に stone の石が 2 つ・空点が 3 つある
    /// 並びの空点を集める。実際に三になるかは打って makes_three で確かめること。
    pub fn three_candidates(&self, stone: Stone) -> Vec<usize> {
        self.window_candidates(stone, 2)
    }

    /// stone の石が置かれた p が、どれかの方向で三 (一手で p を含む達四になれる並び。黒は達四の点が禁じ手なら三でない)
    /// を作っているか。
    pub fn makes_three(&self, stone: Stone, p: usize) -> bool {
        let mut cells = self.cells;
        rules::is_three(&mut cells, p, stone.cell(), self.rules)
    }

    /// 連続する 5 点に stone の石が own 個あり、残りが空点の並びの空点。
    fn window_candidates(&self, stone: Stone, own: usize) -> Vec<usize> {
        let color = stone.cell();
        let mut mark = [false; CELLS];
        for (dr, dc) in [(0isize, 1isize), (1, 0), (1, 1), (1, -1)] {
            for row in 0..SIZE as isize {
                for col in 0..SIZE as isize {
                    let (er, ec) = (row + dr * 4, col + dc * 4);
                    if !(0..SIZE as isize).contains(&er) || !(0..SIZE as isize).contains(&ec) {
                        continue;
                    }
                    let window: [usize; 5] =
                        std::array::from_fn(|k| pos((row + dr * k as isize) as usize, (col + dc * k as isize) as usize));
                    let stones = window.iter().filter(|&&q| self.cells[q] == color).count();
                    let empty = window.iter().filter(|&&q| self.cells[q] == EMPTY).count();
                    if stones == own && stones + empty == 5 {
                        for &q in &window {
                            if self.cells[q] == EMPTY {
                                mark[q] = true;
                            }
                        }
                    }
                }
            }
        }
        (0..CELLS).filter(|&p| mark[p]).collect()
    }

    /// 石を置かずに手番だけを相手に渡す (読みで「一手休んだら」を調べる用)。
    pub fn pass(&mut self) {
        self.side = self.side.opponent();
    }

    /// 石の配置と手番から求めるハッシュ値 (読みの局面表用)。
    pub fn key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.cells.hash(&mut hasher);
        self.side.hash(&mut hasher);
        hasher.finish()
    }

    pub fn check_move(&self, p: usize) -> Result<(), MoveError> {
        if self.result != GameResult::Ongoing {
            return Err(MoveError::GameOver);
        }
        if p >= CELLS {
            return Err(MoveError::OutOfBoard);
        }
        if self.cells[p] != EMPTY {
            return Err(MoveError::Occupied);
        }
        if self.moves == 0 && p != CENTER {
            return Err(MoveError::NotCenter);
        }
        Ok(())
    }

    /// 着手できるか。黒の禁じ手も着手はできる(打つと負けになる)。
    pub fn is_legal(&self, p: usize) -> bool {
        self.check_move(p).is_ok()
    }

    pub fn legal_moves(&self) -> Vec<usize> {
        (0..CELLS).filter(|&p| self.is_legal(p)).collect()
    }

    /// p に打つ。黒が禁じ手を打った場合も石は置かれ、その場で白の勝ちになる。
    pub fn play(&mut self, p: usize) -> Result<GameResult, MoveError> {
        self.check_move(p)?;
        let forbidden = self.side == Stone::Black && self.is_forbidden(p);
        let color = self.side.cell();
        self.cells[p] = color;
        self.moves += 1;
        self.last_move = Some(p);

        if forbidden {
            self.result = GameResult::Win(Stone::White);
        } else if rules::is_five(&self.cells, p, color, self.rules) {
            self.result = GameResult::Win(self.side);
        } else {
            self.side = self.side.opponent();
            if self.moves == CELLS {
                self.result = GameResult::Draw;
            }
        }
        Ok(self.result)
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "   ")?;
        for c in 0..SIZE {
            write!(f, "{c:>2} ")?;
        }
        writeln!(f)?;
        for r in 0..SIZE {
            write!(f, "{r:>2} ")?;
            for c in 0..SIZE {
                let p = pos(r, c);
                let mark = match self.get(p) {
                    Some(Stone::Black) if self.last_move == Some(p) => 'X',
                    Some(Stone::Black) => 'x',
                    Some(Stone::White) if self.last_move == Some(p) => 'O',
                    Some(Stone::White) => 'o',
                    None => '.',
                };
                write!(f, " {mark} ")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
