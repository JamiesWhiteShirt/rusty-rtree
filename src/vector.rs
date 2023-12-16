use std::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, AddAssign, Index, IndexMut, Mul, Sub},
};

use array_init::array_init;

use crate::{
    bounds::{Bounded, Bounds},
    contains::Contains,
    geom::sphere::Sphere,
    intersects::Intersects,
};

#[derive(Clone, Copy, PartialEq)]
pub struct Vector<S, const D: usize>(pub [S; D]);

impl<S, const D: usize> Eq for Vector<S, D> where S: Eq {}

impl<S, const D: usize> Vector<S, D> {
    pub fn zip<'a>(&self, rhs: &'a Self) -> impl ExactSizeIterator<Item = (&S, &'a S)> {
        self.0.iter().zip(rhs.0.iter())
    }

    pub fn into_map<F, U>(self, f: F) -> Vector<U, D>
    where
        F: FnMut(S) -> U,
    {
        Vector(self.0.map(f))
    }

    pub fn map<F, U>(&self, f: F) -> Vector<U, D>
    where
        S: Clone,
        F: FnMut(S) -> U,
    {
        Vector(self.0.clone().map(f))
    }

    pub fn sq_mag(&self) -> S
    where
        S: Clone + Sum + Sub<Output = S> + Mul<Output = S>,
    {
        self.0.iter().map(|v| v.clone() * v.clone()).sum()
    }
}

impl<S, const D: usize> Index<usize> for Vector<S, D> {
    type Output = S;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<S, const D: usize> IndexMut<usize> for Vector<S, D> {
    fn index_mut(&mut self, index: usize) -> &mut S {
        &mut self.0[index]
    }
}

impl<S, const D: usize> Add for Vector<S, D>
where
    S: Add + Clone,
{
    type Output = Vector<S::Output, D>;

    fn add(self, rhs: Self) -> Self::Output {
        Vector(array_init(|i| self.0[i].clone() + rhs.0[i].clone()))
    }
}

impl<S, const D: usize> AddAssign for Vector<S, D>
where
    S: AddAssign + Clone,
{
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..D {
            self.0[i] += rhs.0[i].clone()
        }
    }
}

impl<S, const D: usize> Sub for Vector<S, D>
where
    S: Sub + Clone,
{
    type Output = Vector<S::Output, D>;

    fn sub(self, rhs: Self) -> Self::Output {
        Vector(array_init(|i| self.0[i].clone() - rhs.0[i].clone()))
    }
}

impl<S: Ord, const D: usize> Intersects<Bounds<S, D>> for Vector<S, D> {
    fn intersects(&self, rhs: &Bounds<S, D>) -> bool {
        rhs.contains(self)
    }
}

impl<S: Clone + Sub<Output = S> + Into<f64>, const D: usize> Intersects<Sphere<S, D>>
    for Vector<S, D>
{
    fn intersects(&self, rhs: &Sphere<S, D>) -> bool {
        rhs.intersects(self)
    }
}

impl<S: Ord + Clone, const D: usize> Bounded<S, D> for Vector<S, D> {
    fn bounds(&self) -> Bounds<S, D> {
        Bounds {
            min: self.clone(),
            max: self.clone(),
        }
    }
}

impl<S, const D: usize> Debug for Vector<S, D>
where
    S: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.0.iter()).finish()
    }
}
