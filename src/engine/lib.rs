//! 連珠ルール(黒のみ三三・四四・長連禁止。種類ごとに無効にもできる)の五目並べエンジン。

pub mod board;
pub mod config;
pub mod features;
pub mod inference;
pub mod mcts;
pub mod npz;
pub mod rules;
pub mod selfplay;

#[cfg(test)]
mod tests;

pub use board::{Board, CELLS, CENTER, GameResult, MoveError, SIZE, Stone, pos};
pub use rules::Rules;
pub use mcts::{Evaluation, Evaluator, Mcts, MctsConfig, SearchResult, UniformEvaluator};
