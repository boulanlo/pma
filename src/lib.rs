#![warn(clippy::all)]
use std::ops::Range;

///! A packed-memory array structure implementation

/// An element count bound, representing a minimum and a maximum number of element in
/// a particular place in the PMA.
type Bounds = Range<usize>;

/// A density bounds, representing the allowed density (between 0 and 1) of data in a particular
/// place in the PMA.
type DensityBounds = Range<f64>;

/// An index of a segment in the PMA
type SegmentIndex = usize;

/// A range of segment indexes
type Window = Range<SegmentIndex>;

pub mod index_iterator;
pub mod index_par_iterator;
pub mod merge;
pub mod parallel_merge;
pub mod pma;
pub mod pma_zip;
pub mod sequential_merge;
pub mod subtree_indexable;
pub mod subtree_sizes;
