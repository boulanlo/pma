use crate::{SegmentIndex, SubtreeIndex, SubtreeIndexable, Window};
use std::iter::IntoIterator;

#[derive(Debug)]
pub struct SubtreeSizes(pub(crate) Vec<usize>);

impl SubtreeSizes {
    /// Creates a new SubtreeSizes containing a balanced amount of data.
    ///
    /// # Example
    /// ```
    /// use pma::subtree_sizes::SubtreeSizes;
    ///
    /// let subtree_sizes = SubtreeSizes::new(4, 18);
    ///
    /// assert_eq!(subtree_sizes.into_iter().collect::<Vec<usize>>(), vec![5, 4, 5, 4, 9, 9, 18]);
    /// ```
    pub fn new(segment_count: usize, element_count: usize) -> SubtreeSizes {
        let inner_count = std::iter::repeat(0).take(segment_count * 2 - 1).collect();

        let mut subtree_sizes = SubtreeSizes(inner_count);

        subtree_sizes.update(0, element_count);
        subtree_sizes
    }

    /// Returns the size associated with the subtree.
    ///
    /// # Example
    /// ```
    /// use pma::subtree_sizes::SubtreeSizes;
    ///
    /// let subtree_sizes = SubtreeSizes::new(4, 16);
    ///
    /// assert_eq!(subtree_sizes.get(0), 16);
    ///
    /// assert_eq!(subtree_sizes.get(4), 4);
    /// ```
    pub fn get(&self, subtree_index: SubtreeIndex) -> usize {
        debug_assert!(subtree_index < self.0.len());
        self.0[self.0.len() - subtree_index - 1]
    }

    fn propagate_down(&mut self, subtree_index: SubtreeIndex, element_count: usize) {
        let index = self.0.len() - subtree_index - 1;

        let total_height = (((self.0.len() + 1) / 2) as f64).log2() as usize;
        let height = total_height - ((subtree_index + 1) as f64).log2().floor() as usize;

        self.0[index] = element_count;

        let floor = element_count / 2;
        let ceil = element_count - floor;

        if height != 0 {
            self.propagate_down(subtree_index.left_child(), ceil);
            self.propagate_down(subtree_index.right_child(), floor);
        }
    }

    /// Updates the SubtreeSizes by setting the number of element at a certain
    /// subtree index. The change will be repercuted both downwards and upwards in the
    /// tree.
    ///
    /// # Example
    ///
    /// ```
    /// use pma::subtree_sizes::SubtreeSizes;
    ///
    /// let mut subtree_sizes = SubtreeSizes::new(4, 16);
    ///
    /// subtree_sizes.update(1, 10);
    ///
    /// assert_eq!(subtree_sizes.into_iter().collect::<Vec<usize>>(), vec![4, 4, 5, 5, 8, 10, 18]);
    /// ```
    pub fn update(&mut self, mut subtree_index: SubtreeIndex, element_count: usize) {
        self.propagate_down(subtree_index, element_count);

        while subtree_index != 0 {
            let parent = subtree_index.parent();
            let (left, right) = (parent.left_child(), parent.right_child());

            let index = self.0.len() - parent - 1;
            self.0[index] = self.get(left) + self.get(right);

            subtree_index = parent;
        }
    }

    pub fn slice(&self, window: Window) -> &[usize] {
        &self.0[window]
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &usize> {
        self.0.iter()
    }

    pub fn segment(&self, segment: SegmentIndex) -> usize {
        self.0[segment]
    }

    pub fn size(&self) -> usize {
        self.0.len()
    }
}

impl IntoIterator for SubtreeSizes {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
