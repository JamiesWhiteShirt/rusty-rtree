use std::{array, cmp, ops::Sub};

use crate::{
    bounds::{Bounded, AABB, SAABB},
    intersects::{line_bounds_intersect, Intersects},
    vector::SVec,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Line<N, const D: usize>(pub SVec<N, D>, pub SVec<N, D>);

impl<N, const D: usize> Bounded<SAABB<N, D>> for Line<N, D>
where
    N: Ord + Clone + num_traits::Bounded,
{
    fn bounds(&self) -> SAABB<N, D> {
        let min = SVec(array::from_fn(|i| {
            cmp::min(self.0[i].clone(), self.1[i].clone())
        }));
        let max = SVec(array::from_fn(|i| {
            cmp::max(self.0[i].clone(), self.1[i].clone())
        }));
        AABB { min, max }
    }
}

impl<N, const D: usize> Line<N, D>
where
    N: Clone + Sub<Output = N> + Into<f64>,
{
    pub fn delta(&self) -> SVec<N, D> {
        self.1.clone() - self.0.clone()
    }

    pub fn sq_len(&self) -> f64 {
        (self.delta()).into_map(|n| n.into()).sq_mag()
    }

    pub fn len(&self) -> f64 {
        f64::sqrt(self.sq_len())
    }
}

impl<N, const D: usize> Intersects<SAABB<N, D>> for Line<N, D>
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
{
    fn intersects(&self, rhs: &SAABB<N, D>) -> bool {
        line_bounds_intersect(rhs, &self.0, &self.delta())
    }
}
