use crate::{
    bounds::{Bounded, Bounds},
    intersects::Intersects,
};

pub trait SpatialFilter<N, const D: usize, Key> {
    fn test_bounds(&self, bounds: &Bounds<N, D>) -> bool;
    fn test_key(&self, key: &Key) -> bool;
}

pub struct IntersectionFilter<S> {
    space: S,
}

impl<S> IntersectionFilter<S> {
    pub fn new(space: S) -> IntersectionFilter<S> {
        IntersectionFilter { space }
    }
}

impl<N, const D: usize, S, Key> SpatialFilter<N, D, Key> for IntersectionFilter<S>
where
    N: Ord,
    S: Intersects<Bounds<N, D>>,
    Key: Intersects<S>,
{
    fn test_bounds(&self, bounds: &Bounds<N, D>) -> bool {
        self.space.intersects(bounds)
    }

    fn test_key(&self, key: &Key) -> bool {
        key.intersects(&self.space)
    }
}

pub struct BoundedIntersectionFilter<N, const D: usize, S> {
    bounds: Bounds<N, D>,
    space: S,
}

impl<N, const D: usize, S> BoundedIntersectionFilter<N, D, S>
where
    N: Ord,
    S: Bounded<N, D>,
{
    pub fn new(space: S) -> BoundedIntersectionFilter<N, D, S> {
        BoundedIntersectionFilter {
            bounds: space.bounds(),
            space,
        }
    }
}

impl<N, const D: usize, S, Key> SpatialFilter<N, D, Key> for BoundedIntersectionFilter<N, D, S>
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
