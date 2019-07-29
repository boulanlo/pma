use crate::parallel_merge::ParallelMerge;
use rayon_adaptive::prelude::*;

pub struct SequentialMerge<I: PeekableIterator, J: PeekableIterator> {
    pub(crate) left: I,
    pub(crate) right: J,
    pub(crate) parallel_iterator: *mut ParallelMerge<I, J>,
}

impl<I, J, T> Iterator for SequentialMerge<I, J>
where
    I: PeekableIterator<Item = T>,
    J: PeekableIterator<Item = T>,
    T: Sync + Send + Ord,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let left_slice_is_empty = self.left.is_empty();
        let right_slice_is_empty = self.right.is_empty();
        if !left_slice_is_empty && !right_slice_is_empty {
            if self.left.peek(0) <= self.right.peek(0) {
                self.left.next()
            } else {
                self.right.next()
            }
        } else if right_slice_is_empty {
            self.left.next()
        } else {
            self.right.next()
        }
    }
}

impl<I, J> Drop for SequentialMerge<I, J>
where
    I: PeekableIterator,
    J: PeekableIterator,
{
    fn drop(&mut self) {
        let mut empty_left = self.left.divide_on_left_at(0);
        std::mem::swap(&mut empty_left, &mut self.left);
        let mut empty_right = self.right.divide_on_left_at(0);
        std::mem::swap(&mut empty_right, &mut self.right);
        if let Some(destination) = unsafe { self.parallel_iterator.as_mut() } {
            destination.left = empty_left;
            destination.right = empty_right;
        }
    }
}
