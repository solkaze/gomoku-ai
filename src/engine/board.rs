use std::fmt;

use crate::rules::{self, BLACK, EMPTY, WHITE};

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
    /// 盤が埋まった、または黒に禁じ手以外の着手がない
    Draw,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveError {
    GameOver,
    OutOfBoard,
    Occupied,
    NotCenter,
    Forbidden,
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            MoveError::GameOver => "対局は終了しています",
            MoveError::OutOfBoard => "盤外です",
            MoveError::Occupied => "既に石があります",
            MoveError::NotCenter => "黒の初手は天元 (7, 7) です",
            MoveError::Forbidden => "禁じ手です",
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
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    pub fn new() -> Self {
        Self {
            cells: [EMPTY; CELLS],
            side: Stone::Black,
            moves: 0,
            last_move: None,
            result: GameResult::Ongoing,
        }
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

    /// 空点 p が黒にとって禁じ手か(手番に関係なく判定する)。
    pub fn is_forbidden(&self, p: usize) -> bool {
        let mut cells = self.cells;
        cells[p] == EMPTY && rules::is_forbidden(&mut cells, p)
    }

    /// 黒の禁じ手の点をまとめて求める(手番に関係なく判定する)。
    pub fn forbidden_mask(&self) -> [bool; CELLS] {
        let mut cells = self.cells;
        std::array::from_fn(|p| cells[p] == EMPTY && rules::is_forbidden(&mut cells, p))
    }

    /// stone が打てば五になる空点(黒の五は禁じ手より優先されるので常に合法)。
    pub fn winning_moves(&self, stone: Stone) -> Vec<usize> {
        let color = stone.cell();
        let mut cells = self.cells;
        (0..CELLS)
            .filter(|&p| {
                if cells[p] != EMPTY {
                    return false;
                }
                cells[p] = color;
                let five = rules::is_five(&cells, p, color);
                cells[p] = EMPTY;
                five
            })
            .collect()
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
        if self.side == Stone::Black && self.is_forbidden(p) {
            return Err(MoveError::Forbidden);
        }
        Ok(())
    }

    pub fn is_legal(&self, p: usize) -> bool {
        self.check_move(p).is_ok()
    }

    pub fn legal_moves(&self) -> Vec<usize> {
        (0..CELLS).filter(|&p| self.is_legal(p)).collect()
    }

    fn has_legal_move(&self) -> bool {
        (0..CELLS).any(|p| self.is_legal(p))
    }

    pub fn play(&mut self, p: usize) -> Result<GameResult, MoveError> {
        self.check_move(p)?;
        let color = self.side.cell();
        self.cells[p] = color;
        self.moves += 1;
        self.last_move = Some(p);

        if rules::is_five(&self.cells, p, color) {
            self.result = GameResult::Win(self.side);
        } else {
            self.side = self.side.opponent();
            if !self.has_legal_move() {
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
