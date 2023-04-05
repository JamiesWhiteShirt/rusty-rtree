use std::{cmp::Ordering, ops::Sub};

use noisy_float::types::N64;

use crate::bounds::{min_bounds, Bounded, Bounds};

impl<N: Ord + Copy + Sub<Output = N> + Into<f64>, const D: usize> Bounds<N, D> {
    fn volume_increase_of_min_bounds(&self, other: &Self) -> N64 {
        min_bounds(self, other).volume() - self.volume()
    }
}

pub fn minimal_volume_increase<
    'a,
    N: Ord + Copy + Sub<Output = N> + Into<f64>,
    const D: usize,
    Value: Bounded<N, D>,
>(
    children: &'a mut [Value],
    bounds: &Bounds<N, D>,
) -> Option<&'a mut Value> {
    children.into_iter().min_by(|lhs, rhs| {
        // Optimize for minimal volume increase
        let cmp = N64::cmp(
            &lhs.bounds().volume_increase_of_min_bounds(bounds),
            &rhs.bounds().volume_increase_of_min_bounds(bounds),
        );
        if cmp == Ordering::Equal {
            // If the volume increase is the same, select the child with the smallest volume to start with
            N64::cmp(&lhs.bounds().volume(), &rhs.bounds().volume())
        } else {
            cmp
        }
    })
}
