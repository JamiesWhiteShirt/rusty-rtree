use std::{cmp, ops::Sub};

use array_init::from_iter;
use noisy_float::types::{n64, N64};

pub trait Bounded<N, const D: usize> {
    fn bounds(&self) -> Bounds<N, D>;
}

#[derive(Clone, Copy)]
pub struct Bounds<N, const D: usize> {
    pub min: [N; D],
    pub max: [N; D],
}

pub fn min_bounds<N: Ord + Copy, const D: usize>(
    lhs: &Bounds<N, D>,
    rhs: &Bounds<N, D>,
) -> Bounds<N, D> {
    let min = from_iter(
        lhs.min
            .into_iter()
            .zip(rhs.min)
            .map(|(lhs, rhs)| cmp::min(lhs, rhs)),
    )
    .unwrap();
    let max = from_iter(
        lhs.max
            .into_iter()
            .zip(rhs.max)
            .map(|(lhs, rhs)| cmp::max(lhs, rhs)),
    )
    .unwrap();
    Bounds { min, max }
}

impl<N: Copy, const D: usize> Bounded<N, D> for Bounds<N, D> {
    fn bounds(&self) -> Bounds<N, D> {
        *self
    }
}

impl<N: Ord, const D: usize> Bounds<N, D> {
    pub fn intersects(&self, rhs: &Self) -> bool {
        self.min
            .iter()
            .zip(rhs.max.iter())
            .all(|(lhs_min, rhs_max)| lhs_min <= rhs_max)
            && self
                .max
                .iter()
                .zip(rhs.min.iter())
                .all(|(lhs_max, rhs_min)| lhs_max >= rhs_min)
    }

    pub fn contains(&self, rhs: &Self) -> bool {
        self.min
            .iter()
            .zip(rhs.min.iter())
            .all(|(lhs_min, rhs_min)| lhs_min <= rhs_min)
            && self
                .max
                .iter()
                .zip(rhs.max.iter())
                .all(|(lhs_max, rhs_max)| lhs_max >= rhs_max)
    }
}

impl<N: Copy + Sub<Output = N> + Into<f64>, const D: usize> Bounds<N, D> {
    pub fn volume(&self) -> N64 {
        // TODO: Is there a better way to handle this constraint?
        if D == 0 {
            panic!("Cannot calculate volume of bounds with D = 0")
        }

        self.min
            .iter()
            .zip(self.max.iter())
            .map(|(min, max)| n64((*max - *min).into()))
            .reduce(|acc, length| acc * length)
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::Bounds;

    #[test]
    fn test_intersects() {
        let a: Bounds<i32, 2> = Bounds {
            min: [0, 0],
            max: [2, 2],
        };

        let b: Bounds<i32, 2> = Bounds {
            min: [1, 1],
            max: [3, 3],
        };

        assert_eq!(a.intersects(&b), true);
        assert_eq!(b.intersects(&a), true);
    }
}
