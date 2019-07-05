use std::iter::Filter;
///! A packed-memory array structure implementation
use std::ops::{Index, Range};
use std::vec::IntoIter;

/// A fake Option trait, allowing to store a Some/None information
/// without doubling the size in memory.
///
/// It works by restricting a value in the type's possible values and
/// interpreting it as a None. All other values are Some.
///
/// # Example
/// ```
/// use pma::FakeOption;
///
/// struct Foo<T: FakeOption> {
///     data: Vec<T>,
/// }
///
/// let mut foo: Foo<u64> = Foo {
///     data: vec![u64::none(), 3],
/// };
/// assert!(!foo.data.pop().unwrap().is_none());
/// assert!(foo.data.pop().unwrap().is_none());
/// ```
pub trait FakeOption {
    /// Returns `true` if the value is considered as None, and `false`
    /// otherwise
    ///
    /// # Example
    /// ```
    /// use pma::FakeOption;
    ///
    /// let a = u64::none();
    /// let b = 5u64;
    ///
    /// assert!(a.is_none());
    /// assert!(!b.is_none());
    /// ```
    fn is_none(&self) -> bool;

    /// Returns the None value associated to the type. This value must
    /// be unique and consistent, and should be a seldom used value.
    /// For example, we use `u64::MAX` for the u64 type.
    fn none() -> Self;
}

impl FakeOption for u64 {
    fn is_none(&self) -> bool {
        *self == <u64 as FakeOption>::none()
    }

    fn none() -> Self {
        std::u64::MAX
    }
}

impl FakeOption for u32 {
    fn is_none(&self) -> bool {
        *self == <u32 as FakeOption>::none()
    }

    fn none() -> Self {
        std::u32::MAX
    }
}

pub struct PMASlice<T: FakeOption>([T]);

impl<T: FakeOption> PMASlice<T> {
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter().filter(|&e| !e.is_none())
    }

    pub fn iter_chunks(&self, segment_size: usize) -> impl Iterator<Item = &T> {
        self.0
            .chunks(segment_size)
            .flat_map(|s| s.iter().take_while(|&e| !e.is_none()))
    }
}

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
/// [^1]: Durand, M., Raffin, B. & Faure, F (2012). A Packed Memory Array to Keep Particles Sorted.
pub struct PMA<T: FakeOption> {
    data: Vec<T>,
    pub pma_bounds: Range<usize>,
    pub segment_bounds: Range<usize>,
    segment_size: usize,
}

impl<T: FakeOption> PMA<T> {
    /// Creates a PMA from an `ExactSizeIterator`.
    ///
    /// The current algorithm for creating the PMA guarantees that the density
    /// of the structure will always be 0.5 ± 0.225.
    ///
    /// # Example
    /// ```
    /// use pma::PMA;
    ///
    /// let data = 0u32..1024;
    /// let pma = PMA::from_iterator(data, 0.3..0.7, 0.08..0.92, 8);
    ///
    /// println!("{:?} {:?}", pma.pma_bounds, pma.segment_bounds);
    /// assert!(pma.is_density_respected());
    /// ```
    pub fn from_iterator<I>(
        mut iterator: I,
        pma_bounds: Range<f64>,
        segment_bounds: Range<f64>,
        segment_size: usize,
    ) -> PMA<T>
    where
        I: ExactSizeIterator<Item = T>,
    {
        let element_count = iterator.len();

        let segment_count_upper =
            (((2.0 * element_count as f64) / segment_size as f64) as usize).next_power_of_two();
        let segment_count_lower = segment_count_upper / 2;

        let density_upper = element_count as f64 / (segment_size * segment_count_upper) as f64;
        let density_lower = element_count as f64 / (segment_size * segment_count_lower) as f64;

        let segment_count = if (density_upper - 0.5).abs() < (density_lower - 0.5).abs() {
            segment_count_upper
        } else {
            segment_count_lower
        };

        let mut data: Vec<T> = Vec::new();

        let mod_break = element_count % segment_count;

        for i in 0..segment_count {
            let inserted_count = if i < mod_break {
                (element_count / segment_count) + 1
            } else {
                element_count / segment_count
            };

            for _ in 0..inserted_count {
                data.push(iterator.next().unwrap());
            }
            for _ in 0..segment_size - inserted_count {
                data.push(T::none())
            }
        }

        let array_size = segment_count * segment_size;
        let pma_bounds: Range<usize> = (pma_bounds.start * array_size as f64) as usize
            ..(pma_bounds.end * array_size as f64) as usize;
        let segment_bounds: Range<usize> = (segment_bounds.start * segment_size as f64) as usize
            ..(segment_bounds.end * segment_size as f64) as usize;

        PMA {
            data,
            pma_bounds,
            segment_bounds,
            segment_size,
        }
    }

    /// Returns the number of non-None elements in the structure.
    ///
    /// # Example
    /// ```
    /// use pma::PMA;
    ///
    /// let pma = PMA::from_iterator(0u32..15, 0.3..0.7, 0.08..0.92, 8);
    ///
    /// assert_eq!(pma.element_count(), 15);
    /// ```
    pub fn element_count(&self) -> usize {
        self.data.iter().filter(|e| !e.is_none()).count()
    }

    /// Returns `true` if the density bounds of the overall PMA is respected,
    /// and `false` otherwise.
    pub fn is_density_respected(&self) -> bool {
        let len = self.element_count();
        len <= self.pma_bounds.end && len >= self.pma_bounds.start
    }

    /// Returns the density of the overall PMA
    pub fn density(&self) -> f64 {
        self.element_count() as f64 / self.data.len() as f64
    }

    /// Returns an iterator over valid elements in the PMA, given a range
    ///
    /// Note that the total number of elements iterated over might not match
    /// the range size, as there are gaps in the PMA array.
    ///
    /// # Example
    /// ```
    /// use pma::PMA;
    ///
    /// let pma = PMA::from_iterator(0u32..16, 0.3..0.7, 0.08..0.92, 8);
    /// let expected = (0..16).collect::<Vec<u32>>();
    /// let expected_ref = expected.iter().collect::<Vec<&u32>>();
    ///
    /// assert_eq!(pma.iter(0..32).collect::<Vec<&u32>>(), expected_ref);
    /// ```
    pub fn iter(&self, range: Range<usize>) -> impl Iterator<Item = &T> {
        self.data[range].iter().filter(|&e| !e.is_none())
    }

    /// Same method as `iter()`, but iterates with chunks.
    ///
    /// Note that the total number of elements iterated over might not match
    /// the range size, as there are gaps in the PMA array.
    ///
    /// # Example
    /// ```
    /// use pma::PMA;
    ///
    /// let pma = PMA::from_iterator(0u32..16, 0.3..0.7, 0.08..0.92, 8);
    /// let expected = (0..16).collect::<Vec<u32>>();
    /// let expected_ref = expected.iter().collect::<Vec<&u32>>();
    ///
    /// assert_eq!(pma.iter_chunks(0..32).collect::<Vec<&u32>>(), expected_ref);
    /// ```
    pub fn iter_chunks(&self, range: Range<usize>) -> impl Iterator<Item = &T> {
        self.data[range]
            .chunks(self.segment_size)
            .flat_map(|s| s.iter().take_while(|&e| !e.is_none()))
    }
}

impl<T: FakeOption> Index<Range<usize>> for PMA<T> {
    type Output = [T];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: FakeOption> IntoIterator for PMA<T> {
    type Item = T;
    type IntoIter = Filter<IntoIter<T>, fn(&T) -> bool>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter().filter(|e| !e.is_none())
    }
}
