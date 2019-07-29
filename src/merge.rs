pub struct Merge<'a, T: 'a> {
    pub slices: [&'a [T]; 2],
    pub indexes: [usize; 2],
}

impl<'a, T: 'a> Merge<'a, T> {
    fn advance_on(&mut self, side: usize) -> Option<&'a T> {
        let r = Some(&self.slices[side][self.indexes[side]]);
        self.indexes[side] += 1;
        r
    }
}

impl<'a, T: 'a + Ord> Iterator for Merge<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        let slice1_is_empty = self.indexes[0] >= self.slices[0].len();
        let slice2_is_empty = self.indexes[1] >= self.slices[1].len();
        if !slice1_is_empty && !slice2_is_empty {
            if self.slices[0][self.indexes[0]] <= self.slices[1][self.indexes[1]] {
                self.advance_on(0)
            } else {
                self.advance_on(1)
            }
        } else if !slice1_is_empty {
            self.advance_on(0)
        } else if !slice2_is_empty {
            self.advance_on(1)
        } else {
            None
        }
    }
}
