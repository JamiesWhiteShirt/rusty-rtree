use std::{cmp::Ordering, ops::Sub};

use noisy_float::types::N64;

use crate::bounds::{Bounded, Bounds};

impl<N, const D: usize> Bounds<N, D>
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
{
    fn volume_increase_of_min_bounds(&self, other: &Self) -> N64 {
        Bounds::containing(self, other).volume() - self.volume()
    }
}

pub fn minimal_volume_increase<'a, N, const D: usize, Value>(
    children: impl Iterator<Item = &'a mut Value>,
    bounds: &Bounds<N, D>,
) -> Option<&'a mut Value>
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
    Value: Bounded<N, D>,
{
    children.min_by(|lhs, rhs| {
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
