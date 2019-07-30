use rayon_adaptive::prelude::*;

pub struct PMAZip<I, J> {
    pub(crate) indexed_iterator: I,
    pub(crate) blocked_iterator: J,
}

impl<'a, I, J, T, U> Divisible for PMAZip<I, J>
where
    I: IndexedParallelIterator<Item = T>,
    J: BlockedParallelIterator<Item = U>,
    T: Sync + Send,
    U: Sync + Send,
{
    type Power = rayon_adaptive::BlockedPower;

    fn base_length(&self) -> Option<usize> {
        debug_assert_eq!(
            self.indexed_iterator.base_length(),
            self.blocked_iterator.base_length()
        );

        self.indexed_iterator.base_length()
    }

    fn divide_at(self, index: usize) -> (Self, Self) {
        let (blocked_left, blocked_right) = self.blocked_iterator.divide_at(index);
        let len = blocked_left.base_length().unwrap();

        let (indexed_left, indexed_right) = self.indexed_iterator.divide_at(len);

        debug_assert_eq!(indexed_left.base_length(), blocked_left.base_length());
        debug_assert_eq!(indexed_right.base_length(), blocked_right.base_length());

        (
            PMAZip {
                indexed_iterator: indexed_left,
                blocked_iterator: blocked_left,
            },
            PMAZip {
                indexed_iterator: indexed_right,
                blocked_iterator: blocked_right,
            },
        )
    }
}

impl<I, J, T, U> ParallelIterator for PMAZip<I, J>
where
    I: IndexedParallelIterator<Item = T>,
    J: BlockedParallelIterator<Item = U>,
    T: Sync + Send,
    U: Sync + Send,
{
    type Item = (T, U);
    type SequentialIterator = std::iter::Zip<I::SequentialIterator, J::SequentialIterator>;

    fn to_sequential(self) -> Self::SequentialIterator {
        self.indexed_iterator
            .to_sequential()
            .zip(self.blocked_iterator.to_sequential())
    }

    fn extract_iter(&mut self, size: usize) -> Self::SequentialIterator {
        self.indexed_iterator
            .extract_iter(size)
            .zip(self.blocked_iterator.extract_iter(size))
    }
}
