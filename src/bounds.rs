use std::{cmp, ops::Sub};

use itertools::izip;
use noisy_float::types::{n64, N64};

use crate::{
    contains::Contains,
    geom::{line::Line, ray::Ray},
    intersects::Intersects,
    vector::{SVec, Vector},
};

/// A trait for types that represent bounds. Implementations should be compact
/// and efficient to compute unions and intersections, and may sacrifice
/// accuracy for performance. It is only required that the union of two bounds
/// objects contains both objects.
///
/// Bounds are generally spatial, but may also have non-spatial dimensions.
pub trait Bounds: Sized + Contains<Self> + Intersects<Self> {
    /// Returns an empty bounds object. The bounds object must be the identity
    /// for the union operation.
    fn empty() -> Self;

    /// Returns the union of two bounds objects. The union of two bounds objects
    /// must contain both objects, but may contain any number of other objects.
    fn union(lhs: &Self, rhs: &Self) -> Self;

    /// Returns the union of a collection of bounds objects. The union of a
    /// collection of bounds objects must contain all objects in the collection,
    /// but may contain any number of other objects. If the collection is empty,
    /// the result should be the empty bounds object.
    fn union_all(bounds: impl IntoIterator<Item = Self>) -> Self;
}

pub trait Bounded<B> {
    fn bounds(&self) -> B;
}

pub trait Volume {
    fn volume(&self) -> N64;
}

/// An axis-aligned bounding box defined by a minimum and maximum point.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AABB<V> {
    pub min: V,
    pub max: V,
}

impl<V> Eq for AABB<V> where V: Eq {}

impl<V> Contains<V> for AABB<V>
where
    V: Vector,
{
    fn contains(&self, point: &V) -> bool {
        self.min.all_ge(point) && self.max.all_le(point)
    }
}

impl<V> Contains<AABB<V>> for AABB<V>
where
    V: Vector,
{
    fn contains(&self, bounds: &AABB<V>) -> bool {
        self.min.all_le(&bounds.max) && self.max.all_ge(&bounds.min)
    }
}

impl<V> Intersects<V> for AABB<V>
where
    V: Vector,
{
    fn intersects(&self, point: &V) -> bool {
        self.min.all_le(point) && self.max.all_ge(point)
    }
}

impl<V> Intersects<AABB<V>> for AABB<V>
where
    V: Vector,
{
    fn intersects(&self, bounds: &AABB<V>) -> bool {
        self.min.all_le(&bounds.max) && self.max.all_ge(&bounds.min)
    }
}

impl<V> Bounds for AABB<V>
where
    V: Vector,
{
    fn empty() -> Self {
        AABB {
            min: V::max_value(),
            max: V::min_value(),
        }
    }

    fn union(lhs: &Self, rhs: &Self) -> Self {
        AABB {
            min: lhs.min.componentwise_min(&rhs.min),
            max: lhs.max.componentwise_max(&rhs.max),
        }
    }

    fn union_all(bounds: impl IntoIterator<Item = Self>) -> Self {
        let mut res = AABB::empty();
        for bounds in bounds {
            res = Self::union(&res, &bounds);
        }
        res
    }
}

impl<V> Bounded<AABB<V>> for V
where
    V: Vector + Clone,
{
    fn bounds(&self) -> AABB<V> {
        AABB {
            min: self.clone(),
            max: self.clone(),
        }
    }
}

/// A "simple" axis-aligned bounding box using [`SVec`] for the minimum and
/// maximum points.
pub type SAABB<N, const D: usize> = AABB<SVec<N, D>>;

impl<N, const D: usize> Volume for SAABB<N, D>
where
    N: Clone + Sub<Output = N> + Into<f64>,
{
    fn volume(&self) -> N64 {
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
}

impl<N, const D: usize> SAABB<N, D>
where
    N: Clone + Sub<Output = N> + Into<f64>,
{
    pub fn sq_dist_to(&self, other: &SAABB<N, D>) -> N64 {
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
            let rhs_then_lhs = n64((lhs_min.clone() - rhs_max.clone()).into());

            // The distance between the two bounds is the maximum of the two distances, or 0 if they overlap
            let dist = cmp::max(cmp::max(lhs_then_rhs, rhs_then_lhs), n64(0.0));
            dist * dist
        })
        .reduce(|acc, dist| acc + dist)
        .unwrap()
    }
}

impl<N, const D: usize> Bounded<SAABB<N, D>> for SAABB<N, D>
where
    N: Ord + Clone + num_traits::Bounded,
{
    fn bounds(&self) -> SAABB<N, D> {
        self.clone()
    }
}

impl<N, const D: usize> Intersects<Ray<N, D>> for SAABB<N, D>
where
    N: Clone + Sub<Output = N> + Into<f64>,
{
    fn intersects(&self, rhs: &Ray<N, D>) -> bool {
        rhs.intersects(self)
    }
}

impl<N, const D: usize> Intersects<Line<SVec<N, D>>> for SAABB<N, D>
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
{
    fn intersects(&self, rhs: &Line<SVec<N, D>>) -> bool {
        rhs.intersects(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{intersects::Intersects, vector::SVec};

    use super::SAABB;

    #[test]
    fn test_intersects() {
        let a: SAABB<i32, 2> = SAABB {
            min: SVec([0, 0]),
            max: SVec([2, 2]),
        };

        let b: SAABB<i32, 2> = SAABB {
            min: SVec([1, 1]),
            max: SVec([3, 3]),
        };

        assert_eq!(a.intersects(&b), true);
        assert_eq!(b.intersects(&a), true);
    }
}
