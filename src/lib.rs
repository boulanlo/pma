use std::ops::Range;

pub trait FakeOption {
    fn is_none(&self) -> bool;
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

pub struct PMA<T: FakeOption> {
    data: Vec<T>,
    pma_bounds: Range<f64>,
    segments_bounds: Range<f64>,
    segment_size: usize,
}

impl<T: FakeOption> PMA<T> {
    pub fn from_iterator<I>(
        mut iterator: I,
        pma_bounds: Range<f64>,
        segments_bounds: Range<f64>,
    ) -> PMA<T>
    where
        I: ExactSizeIterator<Item = T>,
    {
        let element_count = iterator.len();
        let segment_size =
            ((2.0 * element_count as f64).log2().ceil() as usize).next_power_of_two();

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

        PMA {
            data,
            pma_bounds,
            segments_bounds,
            segment_size,
        }
    }

    fn element_count(&self) -> usize {
        self.data.iter().filter(|e| !e.is_none()).count()
    }

    fn segment_size(&self) -> usize {
        (2.0 * self.element_count() as f64).log2().ceil() as usize
    }

    fn segment_count(&self) -> usize {
        ((2.0 * self.element_count() as f64) / self.segment_size() as f64).ceil() as usize
    }

    pub fn elements(&self) -> Vec<&T> {
        self.data.iter().filter(|e| !e.is_none()).collect()
    }

    pub fn elements_with_holes(&self) -> Vec<&T> {
        self.data.iter().collect()
    }

    pub fn density(&self) -> f64 {
        self.element_count() as f64 / self.data.len() as f64
    }
}
