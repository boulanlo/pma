extern crate rayon_adaptive;
use crate::sequential_merge::SequentialMerge;
use rayon_adaptive::prelude::*;
use std::iter::Take;

pub struct ParallelMerge<I, J> {
    pub left: I,
    pub right: J,
}

impl<I, J, T> ParallelMerge<I, J>
where
    I: PeekableIterator<Item = T>,
    J: PeekableIterator<Item = T>,
    T: Sync + Send + Ord,
{
    pub fn new(left: I, right: J) -> ParallelMerge<I, J> {
        ParallelMerge { left, right }
    }
}

impl<I, J, T> Divisible for ParallelMerge<I, J>
where
    I: PeekableIterator<Item = T>,
    J: PeekableIterator<Item = T>,
    T: Sync + Send + Ord,
{
    type Power = rayon_adaptive::BlockedPower;

    fn base_length(&self) -> Option<usize> {
        let (left, right) = (
            self.left.base_length().unwrap(),
            self.right.base_length().unwrap(),
        );
        Some(left + right)
    }

    // TODO: Resolve code duplication
    fn divide_at(self, _index: usize) -> (Self, Self) {
        fn dichotomy_search<I, T>(iterator: &I, pivot: &T, length: usize) -> usize
        where
            I: PeekableIterator<Item = T>,
            T: Sync + Send + Ord,
        {
            std::iter::successors(Some(0..length), |r| {
                if r.len() == 0 {
                    None
                } else {
                    let middle = (r.start + r.end) / 2;
                    let element = iterator.peek(middle);
                    if element < *pivot {
                        Some((middle + 1)..r.end)
                    } else if element > *pivot {
                        Some(r.start..middle)
                    } else {
                        Some(middle..middle)
                    }
                }
            })
            .last()
            .unwrap()
            .start
        }

        if self.left.base_length().unwrap() > self.right.base_length().unwrap() {
            if self.right.base_length().unwrap() == 0 {
                let (left_left, left_right) = self.left.divide();
                let (right_left, right_right) = self.right.divide_at(0);
                (
                    ParallelMerge {
                        left: left_left,
                        right: right_left,
                    },
                    ParallelMerge {
                        left: left_right,
                        right: right_right,
                    },
                )
            } else {
                let (left, right) = (self.left, self.right);
                let (big_left, big_right) = left.divide();

                let pivot = big_left.peek(big_left.base_length().unwrap() - 1);
                let len = right.base_length().unwrap();

                let small_index = dichotomy_search(&right, &pivot, len);

                let (small_left, small_right) = right.divide_at(small_index);
                (
                    ParallelMerge {
                        left: big_left,
                        right: small_left,
                    },
                    ParallelMerge {
                        left: big_right,
                        right: small_right,
                    },
                )
            }
        } else {
            if self.left.base_length().unwrap() == 0 {
                let (right_left, right_right) = self.right.divide();
                let (left_left, left_right) = self.left.divide_at(0);
                (
                    ParallelMerge {
                        left: left_left,
                        right: right_left,
                    },
                    ParallelMerge {
                        left: left_right,
                        right: right_right,
                    },
                )
            } else {
                let (left, right) = (self.left, self.right);

                let (big_left, big_right) = right.divide();

                let pivot = big_left.peek(big_left.base_length().unwrap() - 1);
                let len = left.base_length().unwrap();

                let small_index = dichotomy_search(&left, &pivot, len);

                let (small_left, small_right) = left.divide_at(small_index);
                (
                    ParallelMerge {
                        left: small_left,
                        right: big_left,
                    },
                    ParallelMerge {
                        left: small_right,
                        right: big_right,
                    },
                )
            }
        }
    }
}

impl<I, J, T> ParallelIterator for ParallelMerge<I, J>
where
    I: PeekableIterator<Item = T>,
    J: PeekableIterator<Item = T>,
    T: Sync + Send + Ord,
{
    type Item = T;
    type SequentialIterator = Take<SequentialMerge<I, J>>;

    fn to_sequential(self) -> Self::SequentialIterator {
        let left_size = self.left.base_length().unwrap();
        let right_size = self.right.base_length().unwrap();
        SequentialMerge {
            left: self.left,
            right: self.right,
            parallel_iterator: std::ptr::null_mut(),
        }
        .take(left_size + right_size)
    }

    fn extract_iter(&mut self, index: usize) -> Self::SequentialIterator {
        SequentialMerge {
            left: unsafe { std::ptr::read(&self.left as *const I) },
            right: unsafe { std::ptr::read(&self.right as *const J) },
            parallel_iterator: self as *mut ParallelMerge<I, J>,
        }
        .take(index)
    }
}
