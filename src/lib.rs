#![warn(clippy::all)]

///! A packed-memory array structure implementation
pub mod index_iterator;
pub mod index_par_iterator;
pub mod merge;
pub mod parallel_merge;
pub mod pma;
pub mod pma_zip;
pub mod sequential_merge;
