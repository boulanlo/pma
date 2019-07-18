use std::ops::Range;

#[derive(Debug)]
pub struct Window<'a, T> {
    pub(crate) segment_size: usize,
    pub(crate) data: &'a [T],
    pub(crate) sizes: &'a [usize],
    pub(crate) relative_index: usize,
    pub(crate) absolute_index: usize,
    pub(crate) segments_range: Range<usize>,
}

impl<'a, T> Iterator for Window<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.segments_range.end - self.segments_range.start == 0 {
            None
        } else {
            let result = Some(&self.data[self.absolute_index]);
            self.absolute_index += 1;
            self.relative_index += 1;

            if self.sizes[self.segments_range.start] < self.relative_index + 1 {
                self.relative_index = 0;
                self.absolute_index += self.segment_size - self.sizes[self.segments_range.start];
                self.segments_range.start += 1;
            }

            result
        }
    }
}
