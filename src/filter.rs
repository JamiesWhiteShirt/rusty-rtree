use std::marker::PhantomData;

use crate::{bounds::{Bounds, Bounded}, intersects::Intersects};

pub trait SpatialFilter<N: Ord, const D: usize, Key> {
    fn test_bounds(&self, bounds: &Bounds<N, D>) -> bool;
    fn test_key(&self, value: &Key) -> bool;
}

pub struct BoundedIntersectionFilter<N, const D: usize, Key: Intersects<S>, S> {
    bounds: Bounds<N, D>,
    space: S,
    phantom: PhantomData<Key>,
}

impl<N: Ord, const D: usize, Value: Intersects<S>, S: Bounded<N, D>> BoundedIntersectionFilter<N, D, Value, S> {
    pub fn new(space: S) -> BoundedIntersectionFilter<N, D, Value, S> {
        BoundedIntersectionFilter {
            bounds: space.bounds(),
            space,
            phantom: PhantomData,
        }
    }
}

impl<N: Ord, const D: usize, Key: Intersects<S>, S> SpatialFilter<N, D, Key> for BoundedIntersectionFilter<N, D, Key, S> {
    fn test_bounds(&self, bounds: &Bounds<N, D>) -> bool {
        self.bounds.intersects(bounds)
    }

    fn test_key(&self, value: &Key) -> bool {
        value.intersects(&self.space)
    }
}
