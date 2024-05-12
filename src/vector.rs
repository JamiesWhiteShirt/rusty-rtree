use std::{
    array,
    fmt::Debug,
    ops::{Add, AddAssign, Index, IndexMut, Mul, Sub},
};

use crate::{bounds::SAABB, contains::Contains, geom::sphere::Sphere, intersects::Intersects};

pub trait Vector: Sized {
    /// The componentwise minimum value for this type.
    fn min_value() -> Self;
    /// The componentwise maximum value for this type.
    fn max_value() -> Self;

    /// Check if all components of this vector are less than or equal to the corresponding
    /// components of another vector.
    fn all_le(&self, rhs: &Self) -> bool;
    /// Check if all components of this vector are greater than or equal to the corresponding
    /// components of another vector.
    fn all_ge(&self, rhs: &Self) -> bool;

    /// Create a new vector with each component being the minimum of the corresponding components
    /// of this vector and another vector.
    fn componentwise_min(&self, rhs: &Self) -> Self;
    /// Create a new vector with each component being the maximum of the corresponding components
    /// of this vector and another vector.
    fn componentwise_max(&self, rhs: &Self) -> Self;
}

/// A point in D-dimensional space expressed as D scalars of type N. The "S" is short for "simple",
/// referring to the same scalar type being used for all dimensions, with a natural way to express
/// the volume of [SAABB]s created from these points.
#[derive(Clone, Copy, PartialEq, PartialOrd, Hash)]
pub struct SVec<N, const D: usize>(pub [N; D]);

impl<N, const D: usize> Eq for SVec<N, D> where N: Eq {}

impl<N, const D: usize> Vector for SVec<N, D>
where
    N: Clone + Add<Output = N> + Sub<Output = N> + Ord + num_traits::Bounded,
{
    fn min_value() -> Self {
        SVec(array::from_fn(|_| N::min_value()))
    }
    fn max_value() -> Self {
        SVec(array::from_fn(|_| N::max_value()))
    }

    fn all_le(&self, rhs: &Self) -> bool {
        self.0.iter().zip(rhs.0.iter()).all(|(a, b)| a <= b)
    }
    fn all_ge(&self, rhs: &Self) -> bool {
        self.0.iter().zip(rhs.0.iter()).all(|(a, b)| a >= b)
    }

    fn componentwise_min(&self, rhs: &Self) -> Self {
        SVec(array::from_fn(|i| self.0[i].clone().min(rhs.0[i].clone())))
    }
    fn componentwise_max(&self, rhs: &Self) -> Self {
        SVec(array::from_fn(|i| self.0[i].clone().max(rhs.0[i].clone())))
    }
}

impl<N, const D: usize> SVec<N, D> {
    pub fn zip<'a>(&self, rhs: &'a Self) -> impl ExactSizeIterator<Item = (&N, &'a N)> {
        self.0.iter().zip(rhs.0.iter())
    }

    pub fn into_map<F, U>(self, f: F) -> SVec<U, D>
    where
        F: FnMut(N) -> U,
    {
        SVec(self.0.map(f))
    }

    pub fn sq_mag(&self) -> N
    where
        N: Clone + Add<Output = N> + Sub<Output = N> + Mul<Output = N>,
    {
        self.0
            .iter()
            .map(|v| v.clone() * v.clone())
            .reduce(|a, b| a + b)
            .unwrap()
    }
}

impl<N, const D: usize> Index<usize> for SVec<N, D> {
    type Output = N;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<N, const D: usize> IndexMut<usize> for SVec<N, D> {
    fn index_mut(&mut self, index: usize) -> &mut N {
        &mut self.0[index]
    }
}

impl<N, const D: usize> Add for SVec<N, D>
where
    N: Add + Clone,
{
    type Output = SVec<N::Output, D>;

    fn add(self, rhs: Self) -> Self::Output {
        SVec(array::from_fn(|i| self.0[i].clone() + rhs.0[i].clone()))
    }
}

impl<N, const D: usize> AddAssign for SVec<N, D>
where
    N: AddAssign + Clone,
{
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..D {
            self.0[i] += rhs.0[i].clone()
        }
    }
}

impl<N, const D: usize> Sub for SVec<N, D>
where
    N: Sub + Clone,
{
    type Output = SVec<N::Output, D>;

    fn sub(self, rhs: Self) -> Self::Output {
        SVec(array::from_fn(|i| self.0[i].clone() - rhs.0[i].clone()))
    }
}

impl<N: Ord, const D: usize> Intersects<SAABB<N, D>> for SVec<N, D>
where
    SAABB<N, D>: Contains<SVec<N, D>>,
{
    fn intersects(&self, rhs: &SAABB<N, D>) -> bool {
        rhs.contains(self)
    }
}

impl<N: Clone + Sub<Output = N> + Into<f64>, const D: usize> Intersects<Sphere<N, D>>
    for SVec<N, D>
{
    fn intersects(&self, rhs: &Sphere<N, D>) -> bool {
        rhs.intersects(self)
    }
}

impl<N, const D: usize> Debug for SVec<N, D>
where
    N: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.0.iter()).finish()
    }
}
