extern crate rayon_adaptive;
use crate::pma::PMA;
use rayon_adaptive::prelude::*;
use rayon_adaptive::IndexedPower;
use std::ops::Range;

#[derive(Debug)]
pub struct Window<'a, T> {
    pub(crate) pma: &'a PMA<T>,
    pub(crate) segments_range: Range<usize>,
    pub(crate) current_index: usize,
    pub(crate) end_index: usize,
}

impl<'a, T> Iterator for Window<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.segments_range.start == self.segments_range.end
            || (self.segments_range.len() == 1 && self.current_index == self.end_index)
        {
            None
        } else {
            let t = Some(
                &self.pma.data
                    [self.segments_range.start * self.pma.segment_size + self.current_index],
            );
            self.current_index += 1;

            let segment_count = self.pma.data.len() / self.pma.segment_size;

            while self.segments_range.start < segment_count
                && self.current_index >= self.pma.element_counts[self.segments_range.start]
            {
                self.current_index = 0;
                self.segments_range.start += 1;
            }
            t
        }
    }
}

impl<'a, T: std::fmt::Debug> Divisible for Window<'a, T> {
    type Power = IndexedPower;
    fn base_length(&self) -> Option<usize> {
        let range = &self.segments_range;

        let mut length = self.pma.element_counts[range.start..(range.end - 1)]
            .iter()
            .sum();

        length += self.end_index;
        length -= self.current_index;
        Some(length)
    }

    fn divide_at(mut self, index: usize) -> (Self, Self) {
        let (segment_index, sum) = self
            .segments_range
            .clone()
            .scan(-(self.current_index as isize), |current_sum, index| {
                *current_sum += (self.pma.element_counts[index]
                    - if index == self.segments_range.end {
                        self.end_index
                    } else {
                        0
                    }) as isize;
                Some(*current_sum)
            })
            .map(|s| s as usize)
            .enumerate()
            .find(|&(_, s)| s >= index)
            .unwrap();

        let end = (index + self.pma.element_counts[segment_index]) - sum;

        let mut right = Window {
            pma: self.pma,
            segments_range: segment_index..self.segments_range.end,
            current_index: end,
            end_index: self.end_index,
        };

        if self.pma.element_counts[segment_index] == end {
            right.current_index = 0;
            right.segments_range.start += 1;
        }

        self.segments_range.end = segment_index + 1;

        self.end_index = end;

        (self, right)
    }
}
