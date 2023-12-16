pub trait Contains<T> {
    fn contains(&self, rhs: &T) -> bool;
}
