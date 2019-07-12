use crate::window::Window;
use itertools::Itertools;
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

impl<T: Ord + Clone + Default + std::fmt::Debug> PMA<T> {
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

    fn pma_bounds(&self) -> &Range<usize> {
        self.bounds.last().unwrap()
    }

    fn segment_bounds(&self) -> &Range<usize> {
        self.bounds.get(0).unwrap()
    }

    #[inline]
    fn segment_count(&self) -> usize {
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
    pub fn window(&self, window: Range<usize>) -> Window<T> {
        Window {
            pma: &self,
            segments_range: window,
            current_index: 0,
            end_index: self.segment_size,
        }
        //window.into_par_iter().flat_map(move |i| self.segment(i))
    }

    fn find_segment(&self, element: &T) -> usize {
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

        std::cmp::min(result, self.segment_count() - 1)
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
    pub fn elements(&self) -> Window<T> {
        self.window(0..self.segment_count())
    }

    fn check_segment_density(&self, segment: usize, increment: usize) -> Ordering {
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
        dbg!(&window);
        let height = (window.len() as f64).log2() as usize;
        let bounds = self.bounds.get(height).unwrap();
        let size = self.window(window).count() + increment;

        dbg!(size);
        dbg!(bounds);

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

    fn rebalance(&mut self, window: Range<usize>) {
        // idée : prendre en compte l'élément à insérer pour éviter les boucles infinies

        // Get the indexes of all the elements in the window
        let mut indices: Vec<usize> = Vec::new();

        for (size, i) in self.element_counts[window.clone()]
            .iter()
            .zip(window.clone())
        {
            indices.append(&mut (i * self.segment_size..i * self.segment_size + size).collect())
        }

        let slice = self.data.as_mut_slice();

        // Shift all elements of the window on the left
        for (i, j) in indices.iter().enumerate() {
            slice.swap(i, *j);
        }

        let number_of_elements = indices.len();
        let number_of_segments = window.len();

        let elements_per_segment = number_of_elements / number_of_segments;
        let mut leftovers = number_of_elements % number_of_segments;

        let mut new_data_indexes: Vec<usize> = vec![];

        let element_counts_slice = self.element_counts.as_mut_slice();

        // Calculate the new indexes of the element, to be put in the stabilized array
        for i in 0..number_of_segments {
            let start = window.start + (i * self.segment_size);
            let mut end = window.start + (i * self.segment_size) + elements_per_segment;
            if leftovers > 0 {
                end += 1;
                leftovers -= 1;
            }

            element_counts_slice[window.start + i] = (start..end).len();

            new_data_indexes.append(&mut (start..end).collect::<Vec<usize>>())
        }

        let _ = (0..number_of_elements)
            .rev()
            .zip(new_data_indexes.iter().rev())
            .map(|(i, j)| {
                slice.swap(*j, i);
            })
            .collect::<Vec<()>>();
        self.update_window_sizes();
    }

    fn double_size(&mut self) {
        eprintln!("double size");
        let element_count = self.element_counts.last().unwrap();
        let segment_count = self.segment_count() * 2;

        debug_assert_eq!(
            *self.element_counts.last().unwrap(),
            self.elements().count()
        );

        let elements_per_segment = element_count / segment_count;
        let modulo = element_count % segment_count;

        // Iterator over the desired sizes of each segment
        let sizes_iterator = repeat(elements_per_segment + 1)
            .take(modulo)
            .chain(repeat(elements_per_segment).take(segment_count - modulo));

        // Same iterator as sizes_iterator, but used to collect into the new element_counts
        let element_counts: Vec<usize> = repeat(elements_per_segment + 1)
            .take(modulo)
            .chain(repeat(elements_per_segment).take(segment_count - modulo))
            .collect();

        self.element_counts = element_counts;

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

        self.data = data;
    }

    fn update_window_sizes(&mut self) {
        // Calculate the amount of data in each window, from the segment to the whole
        // PMA.
        let height = (self.segment_count() as f64).log2() as usize;

        let window_element_counts: Vec<usize> = self
            .element_counts
            .iter()
            .take(self.segment_count())
            .copied()
            .collect();

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

        let element_counts: Vec<usize> = consecutive_window_counts.into_iter().flatten().collect();
        self.element_counts = element_counts;
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

    // ugly fix for infinite rebalance loop
    fn perform_insert(&mut self, mut element: T, recursion_level: usize) {
        if self.check_pma_density(1) != Ordering::Equal {
            eprintln!("\tPMA density not respected, doubling the size");
            self.double_size();
            self.calculate_bounds();
            self.update_window_sizes();
            self.rebalance(0..self.segment_count());
        }

        let segment = self.find_segment(&element);

        if let Some(window) = self.find_stable_window(segment, 1) {
            if recursion_level == 0 {
                eprintln!(
                    "\tA window under {:?} does not respect density bounds, rebalancing...",
                    window
                );
                self.rebalance(window);
                self.perform_insert(element, recursion_level + 1);
            } else {
                eprintln!("\tRebalance loop detected, doubling the size");
                self.double_size();
                self.calculate_bounds();
                self.update_window_sizes();
                self.rebalance(0..self.segment_count());
            }
        } else {
            eprintln!("\tEvery density bound respected, beginning insertion...");
            // Dichotomy search for position in segment for insertion
            let start = segment * self.segment_size;
            let end = start + self.element_counts.get(segment).unwrap();

            let index: usize = std::iter::successors(Some(start..end), |r| {
                if r.len() == 0 {
                    // Can't use r.is_empty() because of multiple implementations (??)
                    None
                } else {
                    let middle = (r.start + r.end) / 2;
                    if self.data[middle] > element {
                        Some(r.start..(middle - 1))
                    } else if self.data[middle] < element {
                        Some((middle + 1)..r.end)
                    } else {
                        Some(middle..middle)
                    }
                }
            })
            .last()
            .unwrap()
            .start;

            let current_count = *self.element_counts.get(segment).unwrap();

            // Inserting and shifting elements to the right
            for i in index..=(segment * self.segment_size) + current_count {
                std::mem::swap(&mut self.data[i], &mut element);
            }

            self.element_counts.as_mut_slice()[segment] = current_count + 1;
            self.update_window_sizes();

            debug_assert_eq!(
                *self.element_counts.last().unwrap(),
                self.elements().count()
            );
            debug_assert_eq!(self.check_pma_density(0), Ordering::Equal);
        }
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
        eprintln!("Beginning insertion...");

        self.perform_insert(element, 0);
    }

    pub fn insert_bulk(&mut self, elements: Vec<T>) {}
}