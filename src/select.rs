use std::cmp::Ordering;

use noisy_float::types::N64;

use crate::bounds::{Bounded, Bounds, Volume};

pub trait Selector<B> {
    fn select<Value>(
        &mut self,
        children: impl IntoIterator<Item = Value>,
        bounds: &B,
    ) -> Option<Value>
    where
        Value: Bounded<B>;
}

fn volume_increase_of_min_bounds<B>(lhs: &B, rhs: &B) -> N64
where
    B: Bounds + Volume,
{
    Bounds::union(lhs, rhs).volume() - lhs.volume()
}

pub struct MinimalVolumeIncreaseSelector;

impl<B> Selector<B> for MinimalVolumeIncreaseSelector
where
    B: Bounds + Volume,
{
    fn select<Value>(
        &mut self,
        children: impl IntoIterator<Item = Value>,
        bounds: &B,
    ) -> Option<Value>
    where
        Value: Bounded<B>,
    {
        children.into_iter().min_by(|lhs, rhs| {
            // Optimize for minimal volume increase
            let cmp = N64::cmp(
                &volume_increase_of_min_bounds(&lhs.bounds(), bounds),
                &volume_increase_of_min_bounds(&rhs.bounds(), bounds),
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
