use std::{cmp, ops::Sub};

use array_init::from_iter;

use crate::{
    bounds::{Bounded, Bounds},
    intersects::{ray_bounds_intersect, Intersects},
    vector::Vector,
};

pub struct Line<N, const D: usize>(pub Vector<N, D>, pub Vector<N, D>);

impl<N, const D: usize> Bounded<N, D> for Line<N, D>
where
    N: Ord + Clone,
{
    fn bounds(&self) -> Bounds<N, D> {
        let min = Vector(
            from_iter(
                self.0
                    .zip(&self.1)
                    .map(|(start, end)| cmp::min(start, end).clone()),
            )
            .unwrap(),
        );
        let max = Vector(
            from_iter(
                self.0
                    .zip(&self.1)
                    .map(|(start, end)| cmp::max(start, end).clone()),
            )
            .unwrap(),
        );
        Bounds { min, max }
    }
}

impl<N, const D: usize> Line<N, D>
where
    N: Clone + Sub<Output = N> + Into<f64>,
{
    pub fn delta(&self) -> Vector<N, D> {
        self.1.clone() - self.0.clone()
    }

    pub fn sq_len(&self) -> f64 {
        (self.delta()).into_map(|n| n.into()).sq_mag()
    }

    pub fn len(&self) -> f64 {
        f64::sqrt(self.sq_len())
    }
}

impl<N, const D: usize> Intersects<Bounds<N, D>> for Line<N, D>
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
{
    fn intersects(&self, rhs: &Bounds<N, D>) -> bool {
        let mut contained = true;
        // First test that the bounds of the line intersect with the bounds
        for dim in 0..D {
            let dim_min = cmp::min(self.0[dim].clone(), self.1[dim].clone());
            let dim_max = cmp::max(self.0[dim].clone(), self.1[dim].clone());
            if dim_min > rhs.max[dim] || dim_max < rhs.min[dim] {
                return false;
            }

            if dim_min < rhs.min[dim] || dim_max > rhs.max[dim] {
                contained = false;
            }
        }

        // The line is fully contained by the bounds
        if contained {
            return true;
        }

        // If the bounds intersect, the line intersects iff the ray passing
        // through the line also intersects with the bounds
        ray_bounds_intersect(rhs, &self.0, &self.delta())
    }
}
