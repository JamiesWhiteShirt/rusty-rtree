use std::ops::{Add, Sub};

use crate::{
    bounds::{Bounded, Bounds},
    intersects::Intersects,
    vector::Vector,
};

pub struct Sphere<N, const D: usize> {
    pub center: Vector<N, D>,
    pub radius: N,
}

impl<N, const D: usize> Bounded<N, D> for Sphere<N, D>
where
    N: Clone + Sub<Output = N> + Add<Output = N>,
{
    fn bounds(&self) -> Bounds<N, D> {
        Bounds {
            min: self.center.map(|coord| coord - self.radius.clone()),
            max: self.center.map(|coord| coord + self.radius.clone()),
        }
    }
}

impl<N, const D: usize> Intersects<Vector<N, D>> for Sphere<N, D>
where
    N: Clone + Sub<Output = N> + Into<f64>,
{
    fn intersects(&self, rhs: &Vector<N, D>) -> bool {
        let delta: Vector<f64, D> = (rhs.clone() - self.center.clone()).into_map(|n| n.into());
        let radius: f64 = (self.radius.clone()).into();
        delta.sq_mag() <= radius * radius
    }
}