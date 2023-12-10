use std::ops::Sub;

use crate::{
    bounds::Bounds,
    intersects::{ray_bounds_intersect, Intersects},
    vector::Vector,
};

pub struct Ray<N, const D: usize> {
    pub origin: Vector<N, D>,
    pub n: Vector<N, D>,
}

impl<N, const D: usize> Intersects<Bounds<N, D>> for Ray<N, D>
where
    N: Clone + Sub<Output = N> + Into<f64>,
{
    fn intersects(&self, rhs: &Bounds<N, D>) -> bool {
        ray_bounds_intersect(rhs, &self.origin, &self.n)
    }
}
