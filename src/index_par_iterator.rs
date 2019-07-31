use rayon_adaptive::prelude::*;
use rayon_adaptive::IndexedPower;
use std::ops::Range;

pub struct IndexParIterator<'a> {
    pub(crate) element_counts: &'a [usize],
    pub(crate) segment_size: usize,
    pub(crate) window: Range<usize>,
    pub(crate) start_index: usize,
    pub(crate) end_index: usize,
    pub(crate) skipped_elements: usize,
    pub(crate) initial_segment: usize,
    pub(crate) initial_height: usize,
    pub(crate) element_count: usize,
}

impl<'a> IndexParIterator<'a> {
    pub fn new(
        element_counts: &'a [usize],
        segment_size: usize,
        window: Range<usize>,
        number_of_elements: usize,
    ) -> IndexParIterator<'a> {
        let end_index = *element_counts.last().unwrap();

        IndexParIterator {
            element_counts,
            segment_size,
            window: 0..window.len(),
            start_index: 0,
            end_index,
            skipped_elements: 0,
            initial_segment: window.start,
            initial_height: (window.len() as f64).log2() as usize,
            element_count: number_of_elements,
        }
    }

    fn cut_index(
        &self,
        start: usize,
        height: usize,
        element_count: usize,
        index: usize,
    ) -> (usize, usize) {
        if height == 0 {
            (start, index)
        } else {
            let floor = element_count / 2;
            let ceil = element_count - floor;
            if ceil >= index {
                self.cut_index(start, height - 1, ceil, index)
            } else {
                self.cut_index(
                    start + 2usize.pow(height as u32 - 1),
                    height - 1,
                    floor,
                    index - ceil,
                )
            }
        }
    }
}

impl<'a> Divisible for IndexParIterator<'a> {
    type Power = IndexedPower;

    fn base_length(&self) -> Option<usize> {
        debug_assert!(self.window.len() > 0);
        if self.window.len() == 1 {
            Some(self.end_index - self.start_index)
        } else {
            Some(
                self.element_counts[self.window.start..self.window.end - 1]
                    .iter()
                    .sum::<usize>()
                    + self.end_index
                    - self.start_index,
            )
        }
    }

    fn divide_at(mut self, index: usize) -> (Self, Self) {
        debug_assert!(
            self.skipped_elements >= self.element_counts[0..self.window.start].iter().sum()
        );

        let (middle_segment, middle_index) = self.cut_index(
            0,
            self.initial_height,
            self.element_count,
            index + self.skipped_elements,
        );

        let right = IndexParIterator {
            element_counts: self.element_counts,
            segment_size: self.segment_size,
            window: middle_segment..self.window.end,
            start_index: middle_index,
            end_index: self.end_index,
            skipped_elements: self.skipped_elements + index,
            initial_segment: self.initial_segment,
            initial_height: self.initial_height,
            element_count: self.element_count,
        };
        self.window.end = middle_segment + 1;
        self.end_index = middle_index;

        debug_assert!(self.window.len() > 0);
        debug_assert!(right.window.len() > 0);

        debug_assert_eq!(self.base_length(), Some(index));

        (self, right)
    }
}

impl<'a> ParallelIterator for IndexParIterator<'a> {
    type Item = usize;
    type SequentialIterator = std::iter::Take<
        std::iter::Skip<
            std::iter::FlatMap<
                std::iter::Zip<
                    std::iter::Enumerate<std::slice::Iter<'a, usize>>,
                    std::iter::Repeat<(usize, usize)>,
                >,
                std::ops::Range<usize>,
                fn(((usize, &usize), (usize, usize))) -> Range<usize>,
            >,
        >,
    >;

    fn to_sequential(self) -> Self::SequentialIterator {
        fn translate_size(
            ((i, s), (segment_size, initial_segment)): ((usize, &usize), (usize, usize)),
        ) -> Range<usize> {
            (i + initial_segment) * segment_size..((i + initial_segment) * segment_size) + s
        }

        let size = self.element_counts[self.window.clone()]
            .iter()
            .sum::<usize>();

        self.element_counts[self.window.clone()]
            .iter()
            .enumerate()
            .zip(std::iter::repeat((
                self.segment_size,
                self.initial_segment + self.window.start,
            )))
            .flat_map(translate_size as fn(((usize, &usize), (usize, usize))) -> Range<usize>)
            .skip(self.start_index)
            .take(
                size + self.end_index - self.element_counts[self.window.end - 1] - self.start_index,
            )
    }

    fn extract_iter(&mut self, index: usize) -> Self::SequentialIterator {
        self.divide_on_left_at(index).to_sequential()
    }
}
