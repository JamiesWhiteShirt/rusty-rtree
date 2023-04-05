use std::ops::Sub;

use crate::{intersects::Intersects, bounds::{Bounds, Bounded}, sphere::Sphere};

pub trait Positioned<N, const D: usize> {
    fn position(&self) -> [N; D];
}

impl<N: Ord, const D: usize> Intersects<Bounds<N, D>> for [N; D] {
    fn intersects(&self, rhs: &Bounds<N, D>) -> bool {
        rhs.contains_point(self)
    }
}

impl<N: Copy + Sub<Output = N> + Into<f64>, const D: usize> Intersects<Sphere<N, D>> for [N; D] {
    fn intersects(&self, rhs: &Sphere<N, D>) -> bool {
        rhs.intersects(self)
    }
}

impl<N: Ord + Copy, const D: usize> Bounded<N, D> for [N; D] {
    fn bounds(&self) -> Bounds<N, D> {
        Bounds { min: *self, max: *self }
    }
}
