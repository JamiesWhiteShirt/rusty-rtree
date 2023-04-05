pub trait Intersects<T> {
    fn intersects(&self, rhs: &T) -> bool;
}
