use crate::index_par_iterator::IndexParIterator;
use crate::parallel_merge::ParallelMerge;
use crate::pma_zip::PMAZip;
use crate::subtree_indexable::SubtreeIndex;
use crate::subtree_sizes::SubtreeSizes;
use crate::{Bounds, DensityBounds, SegmentIndex, Window};
use itertools::Itertools;
use rayon_adaptive::prelude::*;
use std::cmp::Ordering;
use std::iter::repeat;
use std::ops::Range;

/// A Packed-Memory Array structure implementation, which keeps gaps in the
/// underlying vector to enable fast insertion in the structure.
///
/// ## Structure
/// This structure works by keep a sorted vector of elements, separated into
/// windows of segments. In this implementation, we use a perfect binary tree
/// (implicitly) to keep the number of segments equal to a power of 2. The
/// element vector is sparse : there are gaps in each segments. The density of
/// the segments, each window and the overall PMA is restricted by bounds. The
/// PMA bounds and the segments bounds is fixed by the user (we recommend using
/// 0.3..0.7 for the PMA and 0.08..0.92 for the segments[^1]).
///
/// ## Rebalancing
/// When an element is inserted into the structure, the structure checks if the
/// density bounds are violated. If this is the case, the insertion triggers a
/// _rebalance_ operation, effectively rebalancing the implicit tree to restore
/// the density bounds. If the overall bounds are violated, the structure doubles
/// its vector size and redistributes the elements in the newly extended vector.
///
/// ## Complexity
/// With the gaps scheme, the insertion cost is performed in O(log² N) amortized
/// element moves, and with special patterns (like a random insertion), the cost
/// is reduced to O(log N).[^1]
///
/// [^1]: Durand, M., Raffin, B. & Faure, F (2012). A Packed Memory Array to Keep Particles Sorted
#[derive(Debug)]
pub struct PMA<T> {
    pub data: Vec<T>,
    pub(crate) bounds: Vec<Bounds>,
    pub(crate) segment_size: usize,
    pub(crate) element_counts: SubtreeSizes,
    pma_density_bounds: DensityBounds,
    segment_density_bounds: DensityBounds,
}

impl<T: Ord + Clone + Default + std::fmt::Debug + Sync + Send> PMA<T> {
    /// Creates a PMA from an `ExactSizeIterator`.
    ///
    /// The current algorithm for creating the PMA guarantees that the density
    /// of the structure will always be 0.5 ± 0.225.
    ///
    /// # Example
    /// ```
    /// ```
    pub fn from_iterator<I>(
        iterator: I,
        pma_density_bounds: DensityBounds,
        segment_density_bounds: DensityBounds,
        segment_size: usize,
    ) -> PMA<T>
    where
        I: ExactSizeIterator<Item = T>,
    {
        let total_element_count = iterator.len();

        // Calculate the optimal segment count. The strategy is to try to use two segment counts, and
        // look at the calculated density obtained with each one. We then choose the segment count that
        // yields the density closest to 0.5 (50%).
        let segment_count_upper =
            (((2 * total_element_count) as f64 / segment_size as f64) as usize).next_power_of_two();
        let segment_count_lower = segment_count_upper / 2;

        let density_upper =
            total_element_count as f64 / (segment_size * segment_count_upper) as f64;
        let density_lower =
            total_element_count as f64 / (segment_size * segment_count_lower) as f64;

        let segment_count = if (density_upper - 0.5).abs() < (density_lower - 0.5).abs() {
            segment_count_upper
        } else {
            segment_count_lower
        };

        // Calculate the element counts
        let element_counts = SubtreeSizes::new(segment_count, total_element_count);

        let mut data: Vec<T> = repeat(T::default())
            .take(segment_count * segment_size)
            .collect();

        for (i, e) in element_counts
            .iter()
            .take(segment_count)
            .enumerate()
            .flat_map(|(i, &s)| i * segment_size..i * segment_size + s)
            .zip(iterator)
        {
            data[i] = e;
        }

        // Calculate the PMA and segment bounds from the density bounds
        let pma_size = data.len();
        let pma_bounds: Bounds = (pma_density_bounds.start * pma_size as f64).round() as usize
            ..(pma_density_bounds.end * pma_size as f64).round() as usize;
        let segment_bounds: Bounds = (segment_density_bounds.start * segment_size as f64).round()
            as usize
            ..(segment_density_bounds.end * segment_size as f64).round() as usize;

        // Calculate the window bounds
        let height = ((pma_size / segment_size) as f64).log2() as usize;
        let mut bounds: Vec<Bounds> = Vec::new();

        bounds.push(segment_bounds);

        for i in 1..height {
            let window_size = 2usize.pow(i as u32) * segment_size;
            let start = pma_density_bounds.start
                + ((segment_density_bounds.start - pma_density_bounds.start)
                    * ((height - i) / height) as f64);
            let start = (start * window_size as f64).round() as usize;

            let end = pma_density_bounds.end
                + ((segment_density_bounds.end - pma_density_bounds.end)
                    * ((height - i) / height) as f64);
            let end = (end * window_size as f64).round() as usize;

            bounds.push(start..end);
        }

        bounds.push(pma_bounds);

        PMA {
            data,
            bounds,
            segment_size,
            element_counts,
            pma_density_bounds,
            segment_density_bounds,
        }
    }

    /// Returns the number of non-gap elements in the structure.
    ///
    /// # Example
    /// ```
    /// use pma::pma::PMA;
    ///
    /// let pma = PMA::from_iterator(0u32..32, 0.3..0.7, 0.08..0.92, 8);
    ///
    /// assert_eq!(pma.element_count(), 32);
    /// ```
    pub fn element_count(&self) -> usize {
        self.element_counts.get(0)
    }

    /// Safely get an element from a segment by its index. If the index is invalid
    /// (i.e. is a gap or out of bounds), it will return a None value. Otherwise, the
    /// function will return an option containing the reference to the element.
    ///
    /// # Example
    /// ```
    /// use pma::pma::PMA;
    ///
    /// let pma = PMA::from_iterator(0u32..32, 0.3..0.7, 0.08..0.92, 8);
    ///
    /// assert_eq!(pma.get(0, 2), Some(&2));
    /// assert_eq!(pma.get(0, 7), None);
    /// ```
    pub fn get(&self, segment: SegmentIndex, index: usize) -> Option<&T> {
        if index < self.element_counts.segment(segment) {
            Some(&self.data[segment * self.segment_size + index])
        } else {
            None
        }
    }

    pub fn pma_bounds(&self) -> &Bounds {
        self.bounds.last().unwrap()
    }

    pub fn segment_bounds(&self) -> &Bounds {
        self.bounds.get(0).unwrap()
    }

    #[inline]
    pub fn segment_count(&self) -> usize {
        self.data.len() / self.segment_size
    }

    /// Returns the valid elements inside a segment as an iterator
    ///
    /// # Example
    /// ```
    /// use pma::pma::PMA;
    ///
    /// let pma = PMA::from_iterator(0u32..32, 0.3..0.7, 0.08..0.92, 8);
    ///
    /// let first_segment = pma.segment(0).collect::<Vec<&u32>>();
    ///
    /// assert_eq!(first_segment, [&0, &1, &2, &3]);
    /// ```
    pub fn segment(&self, segment: SegmentIndex) -> impl Iterator<Item = &T> {
        let start = segment * self.segment_size;
        let end = start + self.element_counts.segment(segment);
        (&self.data[start..end]).iter()
    }

    /// Returns an iterator over the valid elements contained in the
    /// window (a range of segments).
    ///
    /// # Example
    /// ```
    /// use pma::pma::PMA;
    ///
    /// let pma = PMA::from_iterator(0u32..32, 0.3..0.7, 0.08..0.92, 8);
    ///
    /// let window = pma.window(0..4).collect::<Vec<&u32>>();
    /// let expected_result = (0..16).collect::<Vec<u32>>();
    ///
    /// assert_eq!(window, expected_result.iter().collect::<Vec<&u32>>());
    /// ```
    pub fn window(&self, window: Window) -> impl Iterator<Item = &T> {
        window
            .flat_map(move |r| {
                r * self.segment_size..(r * self.segment_size) + self.element_counts.segment(r)
            })
            .map(move |i| self.data.get(i).unwrap())
    }

    fn find_segment(&self, element: &T) -> usize {
        if self.segment_count() == 1 {
            0
        } else {
            let result = std::iter::successors(Some(0..self.data.len() / self.segment_size), |r| {
                if r.len() == 0 {
                    None
                } else {
                    let middle = (r.start + r.end) / 2;
                    let segment: Vec<&T> = self.segment(middle).collect();

                    let (first, last) = (segment.get(0).unwrap(), segment.last().unwrap());

                    if *last < element {
                        Some(middle + 1..r.end)
                    } else if *first > element {
                        Some(r.start..if middle == 0 { 0 } else { middle - 1 })
                    } else {
                        Some(middle..middle)
                    }
                }
            })
            .last()
            .unwrap()
            .start;

            std::cmp::min(result, self.segment_count() - 1)
        }
    }

    /// Returns an iterator over the elements in the PMA.
    ///
    /// # Example
    /// ```
    /// use pma::pma::PMA;
    ///
    /// let pma = PMA::from_iterator(0u32..32, 0.3..0.7, 0.08..0.92, 8);
    ///
    /// let mut elements = pma.elements();
    ///
    /// assert_eq!(elements.next(), Some(&0));
    /// assert_eq!(elements.next(), Some(&1));
    ///
    /// let mut elements = elements.skip(29);
    ///
    /// assert_eq!(elements.next(), Some(&31));
    /// assert_eq!(elements.next(), None);
    /// ```
    pub fn elements(&self) -> impl Iterator<Item = &T> {
        self.window(0..self.segment_count())
    }

    pub fn check_segment_density(&self, segment: usize, increment: usize) -> Ordering {
        let bounds = self.segment_bounds();
        let size = self.segment(segment).count() + increment;

        if size < bounds.start {
            Ordering::Less
        } else if size >= bounds.end {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    fn check_subtree_density(&self, subtree_index: SubtreeIndex, increment: usize) -> Ordering {
        let height = (self.bounds.len() - 1) - ((subtree_index + 1) as f64).log2().floor() as usize;
        let bounds = self.bounds.get(height).unwrap();

        let size = self.element_counts.get(subtree_index) + increment;

        if size < bounds.start {
            Ordering::Less
        } else if size > bounds.end {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    fn check_pma_density(&self, increment: usize) -> Ordering {
        self.check_subtree_density(0, increment)
    }

    pub fn pma_density(&self) -> f64 {
        self.element_counts.get(0) as f64 / self.data.len() as f64
    }

    fn find_stable_window(
        &self,
        segment: usize,
        inserted_size: usize,
    ) -> Option<(Window, SubtreeIndex)> {
        // The idea is to use binary logic on the segment index to find the range.
        // Masking each bit from the most significant to the least significant, we get
        // the successive windows to check. For example, if the tree has a height of 3
        // (8 segments, 0 to 7), and the segment n°5 (0b101) doesn't respect the bounds,
        // then we have :
        // - 0bxxx (masking all the bits) gives the range [0..=7]
        // - 0b1xx (masking the second bit) gives the range [4..=7]
        // - 0b10x (masking the LSB) gives the range [5..=6]

        // Precondition : the PMA is stable at the highest level

        let bits = (self.segment_count() as f64).log2() as usize;

        (0..=bits)
            .rev()
            .scan(0, |pos, bit| {
                let old_pos = *pos;
                *pos = old_pos + if bit == 1 { 1 } else { 2 };
                Some((self.element_counts.size() - old_pos - 1, bit))
            })
            .map(|(p, i)| {
                let mut mask = 2usize.pow((bits - i) as u32) - 1;
                mask <<= i;

                let start = segment & mask;
                let end = start + 2usize.pow(i as u32);

                (p, start..end)
            })
            .tuple_windows()
            .find(|(_, (p, _window))| {
                self.check_subtree_density(*p, inserted_size) != Ordering::Equal
            })
            .map(|((p, x), _)| (x, p))
    }

    fn index_par_iterator(
        &self,
        window: Range<usize>,
        number_of_elements: usize,
    ) -> IndexParIterator<'_> {
        IndexParIterator::new(
            &self.element_counts.slice(window.clone()),
            self.segment_size,
            window,
            number_of_elements,
        )
    }

    fn rebalance(&mut self, window: Window, subtree_index: SubtreeIndex) {
        // Get the indexes of all the elements in the window
        let segment_size = self.segment_size;
        let offset = window.start * segment_size;
        let slice = self.data.as_mut_slice();
        let mut number_of_elements = 0;

        // Shift elements on the left
        for (i, j) in self
            .element_counts
            .slice(window.clone())
            .iter()
            .zip(window.clone())
            .flat_map(|(size, i)| i * segment_size..i * segment_size + size)
            .enumerate()
        {
            slice[offset + i] = unsafe { std::ptr::read(&slice[j] as *const T) };
            number_of_elements += 1;
        }

        self.element_counts
            .update(subtree_index, number_of_elements);

        for (i, j) in (0..number_of_elements)
            .rev()
            .zip(self.element_counts.iter().rev())
        {
            slice[*j] = unsafe { std::ptr::read(&slice[i + offset] as *const T) };
        }
    }

    fn double_size(&mut self) {
        let element_count = self.element_count();
        let segment_count = self.segment_count() * 2;
        debug_assert_eq!(self.element_counts.get(0), self.elements().count());

        let element_counts = SubtreeSizes::new(segment_count, element_count);
        let mut data: Vec<T> = Vec::with_capacity(segment_count * self.segment_size);

        unsafe {
            data.set_len(segment_count * self.segment_size);
        }

        for (i, e) in element_counts
            .iter()
            .take(segment_count)
            .enumerate()
            .flat_map(|(i, t)| (i * self.segment_size)..(i * self.segment_size + t))
            .zip(self.elements())
        {
            data[i] = unsafe { std::ptr::read(e as *const T) };
        }

        self.element_counts = element_counts;
        self.data = data;
    }

    fn calculate_bounds(&mut self) {
        let pma_size = self.data.len();
        let pma_bounds: Bounds = (self.pma_density_bounds.start * pma_size as f64) as usize
            ..(self.pma_density_bounds.end * pma_size as f64) as usize;
        let segment_bounds: Bounds = (self.segment_density_bounds.start * self.segment_size as f64)
            as usize
            ..(self.segment_density_bounds.end * self.segment_size as f64) as usize;

        // Calculate the window bounds
        let height = ((pma_size / self.segment_size) as f64).log2() as usize;
        let mut bounds: Vec<Bounds> = Vec::new();

        bounds.push(segment_bounds);

        for i in 1..height {
            let window_size = 2usize.pow(i as u32) * self.segment_size;
            let start = self.pma_density_bounds.start
                + ((self.segment_density_bounds.start - self.pma_density_bounds.start)
                    * ((height - i) / height) as f64);
            let start = (start * window_size as f64) as usize;

            let end = self.pma_density_bounds.end
                + ((self.segment_density_bounds.end - self.pma_density_bounds.end)
                    * ((height - i) / height) as f64);
            let end = (end * window_size as f64) as usize;

            bounds.push(start..end);
        }

        bounds.push(pma_bounds);

        self.bounds = bounds;
    }

    /// Inserts an element in the PMA, while maintaining the density bounds of the structure.
    ///
    /// If the insertion will cause the PMA to fall out of the density bounds, the insertion
    /// will trigger beforehand a rebalance of the first stable window. If no stable window is
    /// found, the whole PMA will double its capacity and trigger a global rebalance. The insertion
    /// will the take place as expected.
    ///
    /// # Example
    /// ```
    /// use pma::pma::PMA;
    ///
    /// let mut pma = PMA::from_iterator(0u32..32, 0.3..0.7, 0.08..0.92, 8);
    ///
    /// assert_eq!(pma.elements().count(), 32);
    ///
    /// pma.insert(23);
    ///
    /// assert_eq!(pma.elements().count(), 33);
    /// ```
    pub fn insert(&mut self, mut element: T) {
        if self.check_pma_density(1) != Ordering::Equal {
            self.double_size();
            self.calculate_bounds();
        }

        let segment = self.find_segment(&element);
        let index = match self
            .segment(segment)
            .cloned()
            .collect::<Vec<T>>()
            .as_slice()
            .binary_search(&element)
        {
            Ok(i) => segment * self.segment_size + i,
            Err(i) => segment * self.segment_size + i,
        };

        let current_count = self.element_counts.segment(segment);
        let subtree_index = self.element_counts.size() - segment;

        // Inserting and shifting elements to the right
        for i in index..=(segment * self.segment_size) + current_count {
            std::mem::swap(&mut self.data[i], &mut element);
        }

        self.element_counts.update(subtree_index, current_count + 1);

        // Rebalance check
        if let Some((window, subtree_index)) = self.find_stable_window(segment, 0) {
            self.rebalance(window, subtree_index);
        }

        debug_assert_eq!(self.check_pma_density(0), Ordering::Equal);
    }

    fn rebalance_insert(&mut self, window: Range<usize>, subtree_index: usize, elements: Vec<T>) {
        let current_elements: Vec<T> = self.window(window.clone()).cloned().collect();

        let element_count = current_elements.len() + elements.len();

        // Update element_counts before index_iterator
        self.element_counts.update(subtree_index, element_count);

        let indexes_iterator =
            self.index_par_iterator(window.clone(), current_elements.len() + elements.len());

        let merge_iterator = ParallelMerge {
            left: current_elements.into_par_iter(),
            right: elements.into_par_iter(),
        };

        debug_assert_eq!(indexes_iterator.base_length(), merge_iterator.base_length());

        let result = PMAZip {
            indexed_iterator: indexes_iterator,
            blocked_iterator: merge_iterator,
        }
        .collect::<Vec<(usize, &T)>>();

        for (destination_index, element) in result {
            self.data.as_mut_slice()[destination_index] = unsafe { std::ptr::read(element) }
        }
    }

    fn perform_insert_bulk(
        &mut self,
        mut elements: Vec<T>,
        window: Range<usize>,
        subtree_index: usize,
    ) {
        // Find the pivot : first element of right window
        if window.len() < 2 {
            debug_assert!(
                elements.len() < self.segment_size - self.element_counts.segment(window.start)
            );

            let segment_size = self.segment_size;
            let mut element_count = 0;

            for (index, element) in (crate::merge::Merge {
                slices: [
                    self.segment(window.start)
                        .cloned()
                        .collect::<Vec<T>>()
                        .as_slice(),
                    elements.as_slice(),
                ],
                indexes: [0, 0],
            })
            .cloned()
            .enumerate()
            .map(|(index, element)| (index + (window.start * segment_size), element))
            {
                self.data.as_mut_slice()[index] = element;
                element_count += 1;
            }

            self.element_counts
                .update(self.element_counts.size() - window.start, element_count);
        } else {
            let pivot_segment_index = (window.len() / 2) + window.start;
            let pivot = self.get(pivot_segment_index, 0).unwrap();

            let element_pivot = match elements.as_slice().binary_search(pivot) {
                Ok(i) => i,
                Err(i) => i,
            };

            let subtree_right = 2 * subtree_index + 1;
            let subtree_left = subtree_right + 1;

            // Check density beforehand
            if self.check_subtree_density(subtree_left, element_pivot) != Ordering::Equal
                || self.check_subtree_density(subtree_right, elements.len() - element_pivot)
                    != Ordering::Equal
            {
                self.rebalance_insert(window, subtree_index, elements);
            } else {
                let right = elements.split_off(element_pivot);
                let left = elements;

                if left.len() > 0 {
                    self.perform_insert_bulk(left, window.start..pivot_segment_index, subtree_left);
                }

                if right.len() > 0 {
                    self.perform_insert_bulk(right, pivot_segment_index..window.end, subtree_right);
                }
            }
        }
    }

    /// Inserts a collection of elements inside the PMA. The collection HAS to be sorted ;
    /// if not, the PMA will loose its sorted aspect.
    ///
    /// # Example
    ///
    /// ```
    /// use pma::pma::PMA;
    /// let a = (0u32..1000).map(|x| 2 * x);
    /// let b = (0u32..1000).map(|x| (2 * x) + 1).collect::<Vec<u32>>();
    ///
    /// let mut pma = PMA::from_iterator(a, 0.3..0.7, 0.08..0.92, 8);
    ///
    /// pma.insert_bulk(b);
    ///
    /// let results = pma.elements().collect::<Vec<&u32>>();
    ///
    /// for x in results.as_slice().windows(2) {
    ///     assert!(x[0] <= x[1]);
    /// }
    /// ```
    pub fn insert_bulk(&mut self, elements: Vec<T>) {
        if self.check_pma_density(elements.len()) != Ordering::Equal {
            // Find appropriate size
            let element_count = self.element_count() + elements.len();
            let pma_size_upper = (element_count * 2).next_power_of_two();
            let pma_size_lower = pma_size_upper / 2;
            let pma_size = if (0.5 - (element_count as f64 / pma_size_upper as f64)).abs()
                < (0.5 - (element_count as f64 / pma_size_lower as f64)).abs()
            {
                pma_size_upper
            } else {
                pma_size_lower
            };

            // Create new data vector
            let mut data: Vec<T> = Vec::with_capacity(pma_size);

            unsafe {
                data.set_len(pma_size);
            }

            let segment_count = pma_size / self.segment_size;

            let current_elements: Vec<T> = self.elements().cloned().collect();

            // Create element count vector
            self.element_counts = SubtreeSizes::new(segment_count, element_count);

            let indexes_iterator =
                self.index_par_iterator(0..(pma_size / self.segment_size), element_count);

            let merge_iterator = ParallelMerge {
                left: current_elements.into_par_iter(),
                right: elements.into_par_iter(),
            };

            let result = PMAZip {
                indexed_iterator: indexes_iterator,
                blocked_iterator: merge_iterator,
            }
            .collect::<Vec<(usize, &T)>>();

            for (destination_index, element) in result {
                data.as_mut_slice()[destination_index] = unsafe { std::ptr::read(element) }
            }

            self.data = data;

            self.calculate_bounds();
        } else {
            self.perform_insert_bulk(elements, 0..self.segment_count(), 0);
        }
        debug_assert!(self.check_pma_density(0) == Ordering::Equal);
    }

    /// Returns true if the searched element is present in the PMA.
    ///
    /// # Example
    /// ```
    /// use pma::pma::PMA;
    ///
    /// let pma = PMA::from_iterator(0u32..32, 0.3..0.7, 0.08..0.92, 8);
    ///
    /// assert!(pma.contains(&5));
    ///
    /// assert!(!pma.contains(&87));
    /// ```
    pub fn contains(&self, element: &T) -> bool {
        let result = std::iter::successors(Some(0..self.data.len() / self.segment_size), |r| {
            if r.len() == 0 {
                None
            } else {
                let middle = (r.start + r.end) / 2;
                let segment: Vec<&T> = self.segment(middle).collect();

                let (first, last) = (segment.get(0).unwrap(), segment.last().unwrap());

                if *last < element {
                    Some(middle + 1..r.end)
                } else if *first > element {
                    Some(r.start..middle - 1)
                } else {
                    Some(middle..middle)
                }
            }
        })
        .last()
        .unwrap()
        .start;

        self.segment(std::cmp::min(result, self.segment_count() - 1))
            .cloned()
            .collect::<Vec<T>>()
            .as_slice()
            .binary_search(element)
            .is_ok()
    }
}
