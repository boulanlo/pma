/// The index of a subtree in the PMA. It is counted backwards, like so :
///                          0
///                       /     \
///                      2       1
///                    /   \   /   \
///                   6     5 4     3
///
/// And so forth.
pub trait SubtreeIndexable {
    fn right_child(&self) -> Self;
    fn left_child(&self) -> Self;
    fn parent(&self) -> Self;
}

pub type SubtreeIndex = usize;

impl SubtreeIndexable for SubtreeIndex {
    fn right_child(&self) -> Self {
        2 * self + 1
    }

    fn left_child(&self) -> Self {
        2 * self + 2
    }

    fn parent(&self) -> Self {
        assert!(*self != 0);

        (self - 1) / 2
    }
}
