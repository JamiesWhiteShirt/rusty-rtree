use std::{cmp::Ordering, ops::Sub};

use noisy_float::types::N64;

use crate::bounds::{Bounded, Bounds, AABB};

pub trait Selector<B> {
    fn select<Value>(
        &mut self,
        children: impl IntoIterator<Item = Value>,
        bounds: &B,
    ) -> Option<Value>
    where
        Value: Bounded<B>;
}

impl<N, const D: usize> AABB<N, D>
where
    N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
{
    fn volume_increase_of_min_bounds(&self, other: &Self) -> N64 {
        Bounds::union(self, other).volume() - self.volume()
    }
}

pub struct MinimalVolumeIncreaseSelector;

impl<N, const D: usize> Selector<AABB<N, D>> for MinimalVolumeIncreaseSelector
where
    N: Ord + Clone + Sub<Output = N> + Into<f64> + num_traits::Bounded,
{
    fn select<Value>(
        &mut self,
        children: impl IntoIterator<Item = Value>,
        bounds: &AABB<N, D>,
    ) -> Option<Value>
    where
        Value: Bounded<AABB<N, D>>,
    {
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
}
