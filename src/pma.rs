use crate::index_iterator::IndexIterator;
use crate::index_par_iterator::IndexParIterator;
use crate::parallel_merge::ParallelMerge;
use crate::pma_zip::PMAZip;
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
    pub(crate) bounds: Vec<Range<usize>>,
    pub(crate) segment_size: usize,
    pub(crate) element_counts: Vec<usize>,
    pma_density_bounds: Range<f64>,
    segment_density_bounds: Range<f64>,
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
        mut iterator: I,
        pma_density_bounds: Range<f64>,
        segment_density_bounds: Range<f64>,
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

        // Arrange data in the vector, with appropriate gaps.
        let element_count_per_segment = total_element_count / segment_count;
        let modulo = total_element_count % segment_count;
        let mut data: Vec<T> = Vec::new();

        let mut window_element_counts: Vec<usize> = Vec::new();

        let mut data_chunk_lengths: Vec<usize> = Vec::new();
        data_chunk_lengths
            .append(&mut repeat(element_count_per_segment + 1).take(modulo).collect());
        data_chunk_lengths.append(
            &mut repeat(element_count_per_segment)
                .take(segment_count - modulo)
                .collect(),
        );

        for chunk_size in data_chunk_lengths {
            let mut chunk: Vec<T> = iterator.by_ref().take(chunk_size).collect();
            window_element_counts.push(chunk_size);

            data.append(&mut chunk);
            data.append(
                &mut repeat(T::default())
                    .take(segment_size - chunk_size)
                    .collect(),
            );
        }

        // Calculate the PMA and segment bounds from the density bounds
        let pma_size = data.len();
        let pma_bounds: Range<usize> = (pma_density_bounds.start * pma_size as f64).round() as usize
            ..(pma_density_bounds.end * pma_size as f64).round() as usize;
        let segment_bounds: Range<usize> = (segment_density_bounds.start * segment_size as f64)
            .round() as usize
            ..(segment_density_bounds.end * segment_size as f64).round() as usize;

        // Calculate the window bounds
        let height = ((pma_size / segment_size) as f64).log2() as usize;
        let mut bounds: Vec<Range<usize>> = Vec::new();

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

        // Calculate the amount of data in each window, from the segment to the whole
        // PMA.
        let mut consecutive_window_counts = vec![window_element_counts];

        for _ in 0..height {
            let window_counts: Vec<usize> = consecutive_window_counts
                .last()
                .unwrap()
                .chunks(2)
                .map(|x| x.iter().sum())
                .collect();
            consecutive_window_counts.push(window_counts);
        }

        let element_counts = consecutive_window_counts.into_iter().flatten().collect();

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
        *self.element_counts.last().unwrap()
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
    pub fn get(&self, segment: usize, index: usize) -> Option<&T> {
        if index < self.element_counts[segment] {
            Some(&self.data[segment * self.segment_size + index])
        } else {
            None
        }
    }

    pub fn pma_bounds(&self) -> &Range<usize> {
        self.bounds.last().unwrap()
    }

    pub fn segment_bounds(&self) -> &Range<usize> {
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
    pub fn segment(&self, segment: usize) -> impl Iterator<Item = &T> {
        let start = segment * self.segment_size;
        let end = start + self.element_counts[segment];
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
    pub fn window(&self, window: Range<usize>) -> impl Iterator<Item = &T> {
        window
            .flat_map(move |r| {
                r * self.segment_size..(r * self.segment_size) + self.element_counts[r]
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

    fn check_window_density(&self, window: Range<usize>, increment: usize) -> Ordering {
        let height = (window.len() as f64).log2() as usize;
        let bounds = self.bounds.get(height).unwrap();

        let size = self.window(window).count() + increment;

        if size < bounds.start {
            Ordering::Less
        } else if size > bounds.end {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    fn check_pma_density(&self, increment: usize) -> Ordering {
        self.check_window_density(0..self.segment_count(), increment)
    }

    pub fn pma_density(&self) -> f64 {
        self.elements().count() as f64 / self.data.len() as f64
    }

    fn find_stable_window(&self, segment: usize, inserted_size: usize) -> Option<Range<usize>> {
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
            .map(|i| {
                let mut mask = 2usize.pow((bits - i) as u32) - 1;
                mask <<= i;

                let start = segment & mask;
                let end = start + 2usize.pow(i as u32);

                start..end
            })
            .tuple_windows()
            .find(|(_, window)| {
                self.check_window_density(window.clone(), inserted_size) != Ordering::Equal
            })
            .map(|(x, _)| x)
    }

    fn index_iterator(
        &self,
        window: Range<usize>,
        number_of_elements: usize,
    ) -> impl Iterator<Item = Range<usize>> {
        let number_of_segments = window.len();
        let segment_size = self.segment_size;
        let height = (window.len() as f64).log2() as usize;

        let elements_per_segment = number_of_elements / number_of_segments;
        let modulo = number_of_elements % number_of_segments;

        window.clone().enumerate().map(move |(i, segment)| {
            let offset = segment * segment_size;
            let reversed = (i as u32).reverse_bits() >> (32 - height);
            let range = offset
                ..offset + elements_per_segment + if reversed < modulo as u32 { 1 } else { 0 };
            range
        })
    }

    fn index_par_iterator(
        &self,
        window: Range<usize>,
        number_of_elements: usize,
    ) -> IndexParIterator<'_, T> {
        IndexParIterator(IndexIterator::new(self, window, number_of_elements))
    }

    fn rebalance(&mut self, window: Range<usize>) {
        // Get the indexes of all the elements in the window
        let segment_size = self.segment_size;
        let offset = window.start * segment_size;
        let slice = self.data.as_mut_slice();
        let mut number_of_elements = 0;

        // Shift elements on the left
        for (i, j) in self.element_counts[window.clone()]
            .iter()
            .zip(window.clone())
            .flat_map(|(size, i)| i * segment_size..i * segment_size + size)
            .enumerate()
        {
            slice[offset + i] = unsafe { std::ptr::read(&slice[j] as *const T) };
            number_of_elements += 1;
        }

        let number_of_segments = window.len();
        let height = (window.len() as f64).log2() as usize;

        let elements_per_segment = number_of_elements / number_of_segments;
        let modulo = number_of_elements % number_of_segments;

        let element_counts_slice = self.element_counts.as_mut_slice();

        // TODO: Find a way to use the index_iterator function without the borrow problems
        let indexes_iterator = (0u32..(number_of_segments as u32))
            .zip(window.clone())
            .flat_map(move |(i, segment)| {
                let offset = segment * segment_size;
                let reversed = i.reverse_bits() >> (32 - height);
                let range = offset
                    ..offset + elements_per_segment + if reversed < modulo as u32 { 1 } else { 0 };
                element_counts_slice[window.start + i as usize] = range.len(); // Update the sizes
                range
            });

        for (i, j) in (0..number_of_elements).rev().zip(indexes_iterator.rev()) {
            slice[j] = unsafe { std::ptr::read(&slice[i + offset] as *const T) };
        }

        self.update_window_sizes();
    }

    fn double_size(&mut self) {
        let element_count = self.element_counts.last().unwrap();
        let segment_count = self.segment_count() * 2;
        debug_assert_eq!(
            *self.element_counts.last().unwrap(),
            self.elements().count()
        );

        let segment_size = self.segment_size;
        let elements_per_segment = element_count / segment_count;
        let modulo = element_count % segment_count;
        let height = (segment_count as f64).log2() as usize;

        let mut element_counts: Vec<usize> = Vec::new();

        // Iterator over the desired sizes of each segment
        let sizes_iterator = (0u32..(segment_count as u32))
            .enumerate()
            .map(|(segment, i)| {
                let offset = segment * segment_size;
                let reversed = i.reverse_bits() >> (32 - height);
                let range = offset
                    ..offset + elements_per_segment + if reversed < modulo as u32 { 1 } else { 0 };
                element_counts.push(range.len());
                range.len()
            });

        let mut data: Vec<T> = Vec::with_capacity(segment_count * self.segment_size);

        unsafe {
            data.set_len(segment_count * self.segment_size);
        }

        for (i, e) in sizes_iterator
            .enumerate()
            .flat_map(|(i, t)| (i * self.segment_size)..(i * self.segment_size + t))
            .zip(self.elements())
        {
            data[i] = unsafe { std::ptr::read(e as *const T) };
        }

        self.element_counts = element_counts;
        self.data = data;
    }

    fn update_window_sizes(&mut self) {
        // Calculate the amount of data in each window, from the segment to the whole
        // PMA.

        let window_element_counts: Vec<usize> =
            self.element_counts[0..self.segment_count()].to_vec();

        self.element_counts = std::iter::successors(Some(window_element_counts), |vec| {
            if vec.len() > 1 {
                Some(
                    vec.chunks(2)
                        .map(|x| x.iter().sum())
                        .collect::<Vec<usize>>(),
                )
            } else {
                None
            }
        })
        .flatten()
        .collect();
    }

    fn calculate_bounds(&mut self) {
        let pma_size = self.data.len();
        let pma_bounds: Range<usize> = (self.pma_density_bounds.start * pma_size as f64) as usize
            ..(self.pma_density_bounds.end * pma_size as f64) as usize;
        let segment_bounds: Range<usize> = (self.segment_density_bounds.start
            * self.segment_size as f64) as usize
            ..(self.segment_density_bounds.end * self.segment_size as f64) as usize;

        // Calculate the window bounds
        let height = ((pma_size / self.segment_size) as f64).log2() as usize;
        let mut bounds: Vec<Range<usize>> = Vec::new();

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
            self.update_window_sizes();
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

        let current_count = *self.element_counts.get(segment).unwrap();

        // Inserting and shifting elements to the right
        for i in index..=(segment * self.segment_size) + current_count {
            std::mem::swap(&mut self.data[i], &mut element);
        }

        self.element_counts.as_mut_slice()[segment] = current_count + 1;
        self.update_window_sizes();

        // Rebalance check
        if let Some(window) = self.find_stable_window(segment, 0) {
            self.rebalance(window);
        }

        debug_assert_eq!(self.check_pma_density(0), Ordering::Equal);
    }

    fn rebalance_insert(&mut self, window: Range<usize>, elements: Vec<T>) {
        let current_elements: Vec<T> = self.window(window.clone()).cloned().collect();

        for (size_index, size) in self
            .index_iterator(window.clone(), current_elements.len() + elements.len())
            .map(|r| r.len())
            .enumerate()
        {
            self.element_counts[window.start + size_index] = size;
        }
        self.update_window_sizes();

        let indexes_iterator =
            self.index_par_iterator(window.clone(), current_elements.len() + elements.len());

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
            self.data.as_mut_slice()[destination_index] = unsafe { std::ptr::read(element) }
        }
    }

    fn perform_insert_bulk(&mut self, mut elements: Vec<T>, window: Range<usize>) {
        // Find the pivot : first element of right window
        if window.len() < 2 {
            debug_assert!(elements.len() < self.segment_size - self.element_counts[window.start]);

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

            self.element_counts[window.start] = element_count;
        } else {
            let pivot_segment_index = (window.len() / 2) + window.start;
            let pivot = self.get(pivot_segment_index, 0).unwrap();

            let element_pivot = match elements.as_slice().binary_search(pivot) {
                Ok(i) => i,
                Err(i) => i,
            };

            // Check density beforehand
            let (window_left, window_right) = (
                window.start..pivot_segment_index,
                pivot_segment_index..window.end,
            );

            if self.check_window_density(window_left, element_pivot) != Ordering::Equal
                || self.check_window_density(window_right, elements.len() - element_pivot)
                    != Ordering::Equal
            {
                self.rebalance_insert(window, elements);
            } else {
                let right = elements.split_off(element_pivot);
                let left = elements;

                if left.len() > 0 {
                    self.perform_insert_bulk(left, window.start..pivot_segment_index);
                }

                if right.len() > 0 {
                    self.perform_insert_bulk(right, pivot_segment_index..window.end);
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
            let element_count = self.element_counts.last().unwrap() + elements.len();
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

            // Create element count vector
            let element_counts: Vec<usize> = self
                .index_iterator(0..(pma_size / self.segment_size), element_count)
                .map(|r| r.len())
                .collect();

            let current_elements: Vec<T> = self.elements().cloned().collect();

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
            self.element_counts = element_counts;

            self.calculate_bounds();
            self.update_window_sizes();
        } else {
            self.perform_insert_bulk(elements, 0..self.segment_count());
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
