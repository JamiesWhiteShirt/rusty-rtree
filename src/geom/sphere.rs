use std::ops::{Add, Sub};

use crate::{
    bounds::{Bounded, SAABB},
    intersects::Intersects,
    vector::SVec,
};

pub struct Sphere<N, const D: usize> {
    pub center: SVec<N, D>,
    pub radius: N,
}

impl<N, const D: usize> Bounded<SAABB<N, D>> for Sphere<N, D>
where
    N: Clone + Sub<Output = N> + Add<Output = N> + Ord + num_traits::Bounded,
{
    fn bounds(&self) -> SAABB<N, D> {
        SAABB {
            min: self
                .center
                .clone()
                .into_map(|coord| coord - self.radius.clone()),
            max: self
                .center
                .clone()
                .into_map(|coord| coord + self.radius.clone()),
        }
    }
}

impl<N, const D: usize> Intersects<SVec<N, D>> for Sphere<N, D>
where
    N: Clone + Sub<Output = N> + Into<f64>,
{
    fn intersects(&self, rhs: &SVec<N, D>) -> bool {
        let delta: SVec<f64, D> = (rhs.clone() - self.center.clone()).into_map(|n| n.into());
        let radius: f64 = (self.radius.clone()).into();
        delta.sq_mag() <= radius * radius
    }
}
