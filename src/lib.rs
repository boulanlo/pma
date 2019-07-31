#![warn(clippy::all)]
use std::ops::Range;

///! A packed-memory array structure implementation

/// An element count bound, representing a minimum and a maximum number of element in
/// a particular place in the PMA.
type Bounds = Range<usize>;

/// A density bounds, representing the allowed density (between 0 and 1) of data in a particular
/// place in the PMA.
type DensityBounds = Range<f64>;

/// The index of a subtree in the PMA. It is counted backwards, like so :
///                          0
///                       /     \
///                      2       1
///                    /   \   /   \
///                   6     5 4     3
///
/// And so forth.
trait SubtreeIndexable {
    fn right_child(&self) -> Self;
    fn left_child(&self) -> Self;
    fn parent(&self) -> Self;
}

type SubtreeIndex = usize;

impl SubtreeIndexable for SubtreeIndex {
    fn right_child(&self) -> Self {
        2 * self + 1
    }

    fn left_child(&self) -> Self {
        2 * self + 2
    }

    fn parent(&self) -> Self {
        assert!(*self != 0);

        (self - 1) / 2
    }
}

type SegmentIndex = usize;
type Window = Range<SegmentIndex>;

pub mod index_iterator;
pub mod index_par_iterator;
pub mod merge;
pub mod parallel_merge;
pub mod pma;
pub mod pma_zip;
pub mod sequential_merge;
pub mod subtree_sizes;
