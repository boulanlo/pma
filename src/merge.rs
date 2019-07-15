use std::iter::Peekable;

pub struct Merge<I: Iterator<Item = T>, J: Iterator<Item = T>, T> {
    left: Peekable<I>,
    right: Peekable<J>,
}

impl<I, J, T> Merge<I, J, T>
where
    I: Iterator<Item = T>,
    J: Iterator<Item = T>,
    T: PartialOrd,
{
    pub fn new(left: I, right: J) -> Merge<I, J, T> {
        Merge {
            left: left.peekable(),
            right: right.peekable(),
        }
    }
}

impl<I, J, T> Iterator for Merge<I, J, T>
where
    I: Iterator<Item = T>,
    J: Iterator<Item = T>,
    T: PartialOrd,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let (l, r) = (self.left.peek(), self.right.peek());

        if l.is_none() && r.is_none() {
            None
        } else if r.is_some() {
            self.right.next()
        } else if l.is_some() {
            self.left.next()
        } else {
            let (l_inner, r_inner) = (l.unwrap(), r.unwrap());
            if l_inner < r_inner {
                self.left.next()
            } else {
                self.right.next()
            }
        }
    }
}
