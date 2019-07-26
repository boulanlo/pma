use crate::parallel_merge::ParallelMerge;
use rayon_adaptive::prelude::*;

pub struct SequentialMerge<I, J> {
    pub(crate) left: Option<I>,
    pub(crate) right: Option<J>,
    pub(crate) counter: usize,
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
        if self.counter == 0 {
            return None;
        }

        self.counter -= 1;

        let (left_iterator, right_iterator) = if self.parallel_iterator.is_null() {
            (self.left.as_mut().unwrap(), self.right.as_mut().unwrap())
        } else {
            unsafe {
                (
                    &mut (*self.parallel_iterator).left,
                    &mut (*self.parallel_iterator).right,
                )
            }
        };

        let (left, right) = (
            left_iterator.base_length().unwrap() > 0,
            right_iterator.base_length().unwrap() > 0,
        );

        if left && right {
            let (left, right) = (left_iterator.peek(0), right_iterator.peek(0));

            if left < right {
                left_iterator.next()
            } else {
                right_iterator.next()
            }
        } else if left {
            left_iterator.next()
        } else if right {
            right_iterator.next()
        } else {
            None
        }
    }
}

/*
impl<I, J> Drop for SequentialMerge<I, J> {
    fn drop(&mut self) {
        if let Some(destination) = unsafe { self.parallel_iterator.as_mut() } {
            unsafe {
                destination.left = std::ptr::read(&self.left as *const I);
                destination.right = std::ptr::read(&self.right as *const J);
            }
            std::mem::forget(self.left);
            std::mem::forget(self.right);
        }
    }
}
*/
