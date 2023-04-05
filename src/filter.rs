use std::marker::PhantomData;

use crate::bounds::{Bounds, Bounded};

pub trait SpatialFilter<N: Ord, const D: usize, Value> {
    fn test_bounds(&self, bounds: &Bounds<N, D>) -> bool;
    fn test_value(&self, value: &Value) -> bool;
}

pub trait Intersectable<T> {
    fn intersects(&self, rhs: &T) -> bool;
}

pub struct BoundedIntersectionFilter<N, const D: usize, Value: Intersectable<S>, S> {
    bounds: Bounds<N, D>,
    space: S,
    phantom: PhantomData<Value>,
}

impl<N: Ord, const D: usize, Value: Intersectable<S>, S: Bounded<N, D>> BoundedIntersectionFilter<N, D, Value, S> {
    pub fn new(space: S) -> BoundedIntersectionFilter<N, D, Value, S> {
        BoundedIntersectionFilter {
            bounds: space.bounds(),
            space,
            phantom: PhantomData,
        }
    }
}

impl<N: Ord, const D: usize, Value: Intersectable<S>, S> SpatialFilter<N, D, Value> for BoundedIntersectionFilter<N, D, Value, S> {
    fn test_bounds(&self, bounds: &Bounds<N, D>) -> bool {
        self.bounds.intersects(bounds)
    }

    fn test_value(&self, value: &Value) -> bool {
        value.intersects(&self.space)
    }
}
