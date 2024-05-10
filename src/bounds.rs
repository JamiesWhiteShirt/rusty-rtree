use std::{array, cmp, ops::Sub};

use array_init::from_iter;
use itertools::izip;
use noisy_float::types::{n64, N64};

use crate::{
    contains::Contains,
    geom::{line::Line, ray::Ray},
    intersects::Intersects,
    vector::Vector,
};

pub trait Bounds: Sized + Contains<Self> {
    fn empty() -> Self;

    fn union(lhs: &Self, rhs: &Self) -> Self;

    fn union_all(bounds: impl IntoIterator<Item = Self>) -> Self;
}

pub trait Bounded<B> {
    fn bounds(&self) -> B;
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AABB<N, const D: usize> {
    pub min: Vector<N, D>,
    pub max: Vector<N, D>,
}

impl<N, const D: usize> Bounds for AABB<N, D>
where
    N: Ord + num_traits::Bounded + Clone,
{
    fn empty() -> AABB<N, D> {
        let min = Vector(array::from_fn(|_| N::max_value()));
        let max = Vector(array::from_fn(|_| N::min_value()));
        AABB { min, max }
    }

    fn union(lhs: &AABB<N, D>, rhs: &AABB<N, D>) -> AABB<N, D> {
        let min = Vector(
            from_iter(
                lhs.min
                    .zip(&rhs.min)
                    .map(|(lhs, rhs)| cmp::min(lhs, rhs).clone()),
            )
            .unwrap(),
        );
        let max = Vector(
            from_iter(
                lhs.max
                    .zip(&rhs.max)
                    .map(|(lhs, rhs)| cmp::max(lhs, rhs).clone()),
            )
            .unwrap(),
        );
        AABB { min, max }
    }

    fn union_all(bounds: impl IntoIterator<Item = AABB<N, D>>) -> AABB<N, D> {
        let mut res = AABB::empty();
        for bounds in bounds {
            res = Self::union(&res, &bounds);
        }
        res
    }
}

impl<N, const D: usize> AABB<N, D> {
    pub fn volume(&self) -> N64
    where
        N: Clone + Sub<Output = N> + Into<f64>,
    {
        // TODO: Is there a better way to handle this constraint?
        if D == 0 {
            panic!("Cannot calculate volume of bounds with D = 0")
        }

        self.min
            .zip(&self.max)
            .map(|(min, max)| n64((max.clone() - min.clone()).into()))
            .reduce(|acc, length| acc * length)
            .unwrap()
    }

    pub fn sq_dist_to(&self, other: &AABB<N, D>) -> N64
    where
        N: Clone + Sub<Output = N> + Into<f64>,
    {
        izip!(
            self.min.0.iter(),
            self.max.0.iter(),
            other.min.0.iter(),
            other.max.0.iter()
        )
        .map(|(lhs_min, lhs_max, rhs_min, rhs_max)| {
            // If lhs < rhs, lhs_then_rhs is a positive number representing the distance between the two
            let lhs_then_rhs = n64((rhs_min.clone() - lhs_max.clone()).into());
            // If lhs > rhs, rhs_then_lhs is a positive number representing the distance between the two
            let rhs_then_lhs = n64((lhs_min.clone() - lhs_max.clone()).into());

            // The distance between the two bounds is the maximum of the two distances, or 0 if they overlap
            let dist = cmp::max(cmp::max(lhs_then_rhs, rhs_then_lhs), n64(0.0));
            dist * dist
        })
        .reduce(|acc, dist| acc + dist)
        .unwrap()
    }
}

impl<N, const D: usize> Eq for AABB<N, D> where N: Eq {}

impl<N, const D: usize> Bounded<AABB<N, D>> for AABB<N, D>
where
    N: Ord + Clone + num_traits::Bounded,
{
    fn bounds(&self) -> AABB<N, D> {
        self.clone()
    }
}

impl<N, const D: usize> Intersects<AABB<N, D>> for AABB<N, D>
where
    N: Ord,
{
    fn intersects(&self, rhs: &AABB<N, D>) -> bool {
        self.min
            .zip(&rhs.max)
            .all(|(lhs_min, rhs_max)| lhs_min <= rhs_max)
            && self
                .max
                .zip(&rhs.min)
                .all(|(lhs_max, rhs_min)| lhs_max >= rhs_min)
    }
}

impl<N, const D: usize> Intersects<Vector<N, D>> for AABB<N, D>
where
    N: Ord,
{
    fn intersects(&self, rhs: &Vector<N, D>) -> bool {
        self.min
            .zip(rhs)
            .all(|(lhs_min, rhs_min)| lhs_min <= rhs_min)
            && self
                .max
                .zip(rhs)
                .all(|(lhs_max, rhs_max)| lhs_max >= rhs_max)
    }
}

impl<N, const D: usize> Intersects<Ray<N, D>> for AABB<N, D>
where
    N: Clone + Sub<Output = N> + Into<f64>,
{
    fn intersects(&self, rhs: &Ray<N, D>) -> bool {
        rhs.intersects(self)
    }
}

impl<N, const D: usize> Intersects<Line<N, D>> for AABB<N, D>
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
{
    fn intersects(&self, rhs: &Line<N, D>) -> bool {
        rhs.intersects(self)
    }
}

impl<N, const D: usize> Contains<AABB<N, D>> for AABB<N, D>
where
    N: Ord,
{
    fn contains(&self, rhs: &Self) -> bool {
        self.min
            .zip(&rhs.min)
            .all(|(lhs_min, rhs_min)| lhs_min <= rhs_min)
            && self
                .max
                .zip(&rhs.max)
                .all(|(lhs_max, rhs_max)| lhs_max >= rhs_max)
    }
}

impl<N, const D: usize> Contains<Vector<N, D>> for AABB<N, D>
where
    N: Ord,
{
    fn contains(&self, point: &Vector<N, D>) -> bool {
        self.intersects(point)
    }
}

#[cfg(test)]
mod tests {
    use crate::{intersects::Intersects, vector::Vector};

    use super::AABB;

    #[test]
    fn test_intersects() {
        let a: AABB<i32, 2> = AABB {
            min: Vector([0, 0]),
            max: Vector([2, 2]),
        };

        let b: AABB<i32, 2> = AABB {
            min: Vector([1, 1]),
            max: Vector([3, 3]),
        };

        assert_eq!(a.intersects(&b), true);
        assert_eq!(b.intersects(&a), true);
    }
}
