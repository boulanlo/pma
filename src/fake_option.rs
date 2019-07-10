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
