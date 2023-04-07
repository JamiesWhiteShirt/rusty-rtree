use std::marker::PhantomData;

use crate::{
    bounds::{Bounded, Bounds},
    intersects::Intersects,
};

pub trait SpatialFilter<N, const D: usize, Key> {
    fn test_bounds(&self, bounds: &Bounds<N, D>) -> bool;
    fn test_key(&self, key: &Key) -> bool;
}

pub struct IntersectionFilter<Key, S> {
    space: S,
    key_phantom: PhantomData<Key>,
}

impl<Key, S> IntersectionFilter<Key, S> {
    pub fn new(space: S) -> IntersectionFilter<Key, S> {
        IntersectionFilter {
            space,
            key_phantom: PhantomData,
        }
    }
}

impl<N, const D: usize, Key, S> SpatialFilter<N, D, Key> for IntersectionFilter<Key, S>
where
    N: Ord,
    Key: Intersects<S>,
    S: Intersects<Bounds<N, D>>,
{
    fn test_bounds(&self, bounds: &Bounds<N, D>) -> bool {
        self.space.intersects(bounds)
    }

    fn test_key(&self, key: &Key) -> bool {
        key.intersects(&self.space)
    }
}

pub struct BoundedIntersectionFilter<N, const D: usize, Key, S> {
    bounds: Bounds<N, D>,
    space: S,
    phantom: PhantomData<Key>,
}

impl<N, const D: usize, Value, S> BoundedIntersectionFilter<N, D, Value, S>
where
    N: Ord,
    Value: Intersects<S>,
    S: Bounded<N, D>,
{
    pub fn new(space: S) -> BoundedIntersectionFilter<N, D, Value, S> {
        BoundedIntersectionFilter {
            bounds: space.bounds(),
            space,
            phantom: PhantomData,
        }
    }
}

impl<N, const D: usize, Key, S> SpatialFilter<N, D, Key> for BoundedIntersectionFilter<N, D, Key, S>
where
    N: Ord,
    Key: Intersects<S>,
{
    fn test_bounds(&self, bounds: &Bounds<N, D>) -> bool {
        self.bounds.intersects(bounds)
    }

    fn test_key(&self, key: &Key) -> bool {
        key.intersects(&self.space)
    }
}
