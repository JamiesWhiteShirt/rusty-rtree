use std::{
    array, cmp,
    ops::{Index, Sub},
};

use array_init::from_iter;
use noisy_float::types::{n64, N64};

use crate::{
    contains::Contains, geom::line::Line, geom::ray::Ray, intersects::Intersects, vector::Vector,
};

pub trait Bounded<N, const D: usize> {
    fn bounds(&self) -> Bounds<N, D>;
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Bounds<N, const D: usize> {
    pub min: Vector<N, D>,
    pub max: Vector<N, D>,
}

impl<N, const D: usize> Bounds<N, D> {
    pub fn empty() -> Bounds<N, D>
    where
        N: num_traits::Bounded,
    {
        let min = Vector(array::from_fn(|_| N::max_value()));
        let max = Vector(array::from_fn(|_| N::min_value()));
        Bounds { min, max }
    }

    pub fn union(lhs: &Bounds<N, D>, rhs: &Bounds<N, D>) -> Bounds<N, D>
    where
        N: Ord + Clone,
    {
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
        Bounds { min, max }
    }

    pub fn union_all(bounds: impl IntoIterator<Item = Bounds<N, D>>) -> Bounds<N, D>
    where
        N: Ord + num_traits::Bounded + Clone,
    {
        let mut res = Bounds::empty();
        for bounds in bounds {
            res = Self::union(&res, &bounds);
        }
        res
    }
}

impl<N, const D: usize> Eq for Bounds<N, D> where N: Eq {}

impl<N, const D: usize> Bounded<N, D> for Bounds<N, D>
where
    N: Clone,
{
    fn bounds(&self) -> Bounds<N, D> {
        self.clone()
    }
}

impl<N, const D: usize> Intersects<Bounds<N, D>> for Bounds<N, D>
where
    N: Ord,
{
    fn intersects(&self, rhs: &Bounds<N, D>) -> bool {
        self.min
            .zip(&rhs.max)
            .all(|(lhs_min, rhs_max)| lhs_min <= rhs_max)
            && self
                .max
                .zip(&rhs.min)
                .all(|(lhs_max, rhs_min)| lhs_max >= rhs_min)
    }
}

impl<N, const D: usize> Intersects<Vector<N, D>> for Bounds<N, D>
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

impl<N, const D: usize> Intersects<Ray<N, D>> for Bounds<N, D>
where
    N: Clone + Sub<Output = N> + Into<f64>,
{
    fn intersects(&self, rhs: &Ray<N, D>) -> bool {
        rhs.intersects(self)
    }
}

impl<N, const D: usize> Intersects<Line<N, D>> for Bounds<N, D>
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
{
    fn intersects(&self, rhs: &Line<N, D>) -> bool {
        rhs.intersects(self)
    }
}

impl<N, const D: usize> Contains<Bounds<N, D>> for Bounds<N, D>
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

impl<N, const D: usize> Contains<Vector<N, D>> for Bounds<N, D>
where
    N: Ord,
{
    fn contains(&self, point: &Vector<N, D>) -> bool {
        self.intersects(point)
    }
}

impl<N, const D: usize> Bounds<N, D>
where
    N: Clone + Sub<Output = N> + Into<f64>,
{
    pub fn volume(&self) -> N64 {
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

impl<N, const D: usize> Index<usize> for Bounds<N, D> {
    type Output = Vector<N, D>;

    fn index(&self, index: usize) -> &Self::Output {
        [&self.min, &self.max][index]
    }
}

#[cfg(test)]
mod tests {
    use crate::{intersects::Intersects, vector::Vector};

    use super::Bounds;

    #[test]
    fn test_intersects() {
        let a: Bounds<i32, 2> = Bounds {
            min: Vector([0, 0]),
            max: Vector([2, 2]),
        };

        let b: Bounds<i32, 2> = Bounds {
            min: Vector([1, 1]),
            max: Vector([3, 3]),
        };

        assert_eq!(a.intersects(&b), true);
        assert_eq!(b.intersects(&a), true);
    }
}
