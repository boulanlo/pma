extern crate either;
use crate::pma::PMA;
use either::Either;
use std::ops::Range;

pub struct IndexIterator<'a, T> {
    pub(crate) pma: &'a PMA<T>,
    pub(crate) window: Range<usize>,
    pub(crate) repartition: Vec<usize>,
    pub(crate) current_element: usize,
    pub(crate) current_segment: usize,
    pub(crate) end_index: usize,
}

impl<'a, T> IndexIterator<'a, T> {
    pub fn new(
        pma: &'a PMA<T>,
        window: Range<usize>,
        number_of_elements: usize,
    ) -> IndexIterator<'a, T> {
        let height = (window.len() as f64).log2() as usize;
        let modulo = number_of_elements % window.len();

        let repartition = window
            .clone()
            .enumerate()
            .map(|(i, _)| {
                (number_of_elements / window.len())
                    + if ((i as u32).reverse_bits() >> (32 - height)) < modulo as u32 {
                        1
                    } else {
                        0
                    }
            })
            .collect::<Vec<usize>>();

        let end_index = *repartition.last().unwrap();

        IndexIterator {
            pma,
            window: window.clone(),
            repartition,
            current_element: 0,
            current_segment: 0,
            end_index,
        }
    }
    pub fn segments_number(&self) -> usize {
        self.window.len()
    }

    pub fn remaining_segment_sizes(&self) -> impl Iterator<Item = usize> + '_ {
        if self.segments_number() == 1 {
            Either::Left(std::iter::once(self.end_index - self.current_element))
        } else {
            Either::Right(
                self.repartition
                    .first()
                    .map(|f| f - self.current_element)
                    .into_iter()
                    .chain(
                        (1..(self.segments_number() - 1))
                            .map(move |i| self.repartition[self.current_segment + i]),
                    )
                    .chain(std::iter::once(self.end_index)),
            )
        }
    }
}

impl<'a, T> Iterator for IndexIterator<'a, T> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.window.len() == 0
            || (self.window.len() == 1 && self.current_element == self.end_index)
        {
            None
        } else {
            let offset = self.window.start * self.pma.segment_size;
            let result = self.current_element + offset;

            self.current_element += 1;

            if self.current_element >= *self.repartition.get(self.current_segment).unwrap() {
                self.current_element = 0;
                self.current_segment += 1;
                self.window.start += 1;
            }

            Some(result)
        }
    }
}
