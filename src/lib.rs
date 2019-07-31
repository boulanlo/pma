#![warn(clippy::all)]
use std::ops::Range;

///! A packed-memory array structure implementation

/// An element count bound, representing a minimum and a maximum number of element in
/// a particular place in the PMA.
type Bounds = Range<usize>;

pub mod index_iterator;
pub mod index_par_iterator;
pub mod merge;
pub mod parallel_merge;
pub mod pma;
pub mod pma_zip;
pub mod sequential_merge;
