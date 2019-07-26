use crate::index_iterator::IndexIterator;
use rayon_adaptive::prelude::*;
use rayon_adaptive::IndexedPower;

pub struct IndexParIterator<'a, T>(pub IndexIterator<'a, T>);

impl<'a, T> Divisible for IndexParIterator<'a, T>
where
    T: Sync,
{
    type Power = IndexedPower;

    fn base_length(&self) -> Option<usize> {
        if self.0.window.len() == 1 {
            Some(self.0.end_index - self.0.current_element)
        } else {
            Some(
                self.0.repartition[self.0.window.start..self.0.window.end - 1]
                    .iter()
                    .sum::<usize>()
                    + self.0.end_index
                    - self.0.current_element,
            )
        }
    }

    fn divide_at(mut self, mut index: usize) -> (Self, Self) {
        debug_assert!(index <= self.base_length().unwrap());
        let initial_size = self.base_length();
        let initia_index = index;

        index += self.0.current_element;
        let (segment_index, cumulated_size) = self.0.repartition[self.0.window.clone()]
            .iter()
            .scan(0, |c, s| {
                *c += s;
                Some(*c)
            })
            .enumerate()
            .find(|&(_, s)| s > index)
            .unwrap_or((0, index));

        let new_current_segment = self.0.current_segment + segment_index;
        let new_current_element = self.0.repartition[new_current_segment] + index - cumulated_size;

        let new_iterator = IndexIterator {
            pma: self.0.pma,
            window: new_current_segment..self.0.window.end,
            repartition: self.0.repartition.clone(),
            current_element: new_current_element,
            current_segment: new_current_segment,
            end_index: self.0.end_index,
        };

        self.0.window.end = new_current_segment + 1;
        self.0.end_index = new_current_element;

        assert!(new_current_element <= self.0.repartition[new_current_segment]);
        let r = (self, IndexParIterator(new_iterator));

        assert_eq!(r.0.base_length(), Some(initia_index));
        assert_eq!(r.1.base_length(), initial_size.map(|s| s - initia_index));

        r
    }
}

impl<'a, T> ParallelIterator for IndexParIterator<'a, T>
where
    T: Sync + Send,
{
    type Item = usize;
    type SequentialIterator = IndexIterator<'a, T>;

    fn to_sequential(self) -> Self::SequentialIterator {
        self.0
    }

    fn extract_iter(&mut self, index: usize) -> Self::SequentialIterator {
        self.divide_on_left_at(index).0
    }
}
