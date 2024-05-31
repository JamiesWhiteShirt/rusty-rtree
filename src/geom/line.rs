use std::ops::Sub;

use crate::{
    bounds::{Bounded, AABB, SAABB},
    intersects::{line_bounds_intersect, Intersects},
    vector::{SVec, Vector},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Line<V>(pub V, pub V);

impl<V> Bounded<AABB<V>> for Line<V>
where
    V: Vector,
{
    fn bounds(&self) -> AABB<V> {
        AABB {
            min: self.0.componentwise_min(&self.1),
            max: self.0.componentwise_max(&self.1),
        }
    }
}

impl<V> Line<V>
where
    V: Sub<Output = V> + Clone,
{
    pub fn delta(&self) -> V {
        self.1.clone() - self.0.clone()
    }
}

impl<N, const D: usize> Line<SVec<N, D>>
where
    N: Clone + Sub<Output = N> + Into<f64>,
{
    pub fn sq_len(&self) -> f64 {
        (self.delta()).into_map(|n| n.into()).sq_mag()
    }

    pub fn len(&self) -> f64 {
        f64::sqrt(self.sq_len())
    }
}

impl<N, const D: usize> Intersects<SAABB<N, D>> for Line<SVec<N, D>>
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
{
    fn intersects(&self, rhs: &SAABB<N, D>) -> bool {
        line_bounds_intersect(rhs, &self.0, &self.delta())
    }
}
