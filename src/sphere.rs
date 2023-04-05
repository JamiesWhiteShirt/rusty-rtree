use std::ops::{Sub, Add};

use crate::{intersects::Intersects, bounds::{Bounded, Bounds}};

pub struct Sphere<N, const D: usize> {
    pub center: [N; D],
    pub radius: N,
}

impl<N: Copy + Sub<Output = N> + Add<Output = N>, const D: usize> Bounded<N, D> for Sphere<N, D> {
    fn bounds(&self) -> Bounds<N, D> {
        Bounds {
            min: self.center.map(|coord| coord - self.radius),
            max: self.center.map(|coord| coord + self.radius),
        }
    }
}

impl<N: Copy + Sub<Output = N> + Into<f64>, const D: usize> Intersects<[N; D]> for Sphere<N, D> {
    fn intersects(&self, rhs: &[N; D]) -> bool {
        let sq_dist = self.center
            .iter()
            .zip(rhs.iter())
            .map(|(center_coord, pos_coord)| {
                let comp_dist = (*pos_coord - *center_coord).into();
                comp_dist * comp_dist
            })
            .reduce(|acc, length| acc * length)
            .unwrap();
        sq_dist <= self.radius.into() * self.radius.into()
    }
}
