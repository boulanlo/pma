use crate::parallel_merge::ParallelMerge;
use rayon_adaptive::prelude::*;

pub struct SequentialMerge<I: PeekableIterator, J: PeekableIterator> {
    pub(crate) left: I,
    pub(crate) right: J,
    pub(crate) left_size: usize,
    pub(crate) right_size: usize,
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
        if self.left_size != 0 && self.right_size != 0 {
            if self.left.peek(0) <= self.right.peek(0) {
                self.left_size -= 1;
                self.left.next()
            } else {
                self.right_size -= 1;
                self.right.next()
            }
        } else if self.left_size != 0 {
            // no need to decrement counter: we always come here anyway
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
