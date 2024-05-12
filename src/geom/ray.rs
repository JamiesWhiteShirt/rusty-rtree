use std::ops::Sub;

use crate::{
    bounds::SAABB,
    intersects::{ray_bounds_intersect, Intersects},
    vector::SVec,
};

pub struct Ray<N, const D: usize> {
    pub origin: SVec<N, D>,
    pub n: SVec<N, D>,
}

impl<N, const D: usize> Intersects<SAABB<N, D>> for Ray<N, D>
where
    N: Clone + Sub<Output = N> + Into<f64>,
{
    fn intersects(&self, rhs: &SAABB<N, D>) -> bool {
        ray_bounds_intersect(rhs, &self.origin, &self.n)
    }
}
