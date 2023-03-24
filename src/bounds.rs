use std::{
    cmp,
    ops::{Mul, Sub},
};

use array_init::from_iter;

trait Bounded<N: Ord, const D: usize> {
    fn bounds(&self) -> Bounds<N, D>;
}

#[derive(Clone, Copy)]
struct Bounds<N: Ord, const D: usize> {
    min: [N; D],
    max: [N; D],
}

fn min_bounds<N: Ord + Copy, const D: usize>(
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

impl<N: Ord + Copy, const D: usize> Bounded<N, D> for Bounds<N, D> {
    fn bounds(&self) -> Bounds<N, D> {
        *self
    }
}

impl<N: Ord, const D: usize> Bounds<N, D> {
    fn intersects(&self, rhs: &Self) -> bool {
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

    fn contains(&self, rhs: &Self) -> bool {
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

impl<N: Ord + Copy + Sub<Output = N> + Mul<Output = N>, const D: usize> Bounds<N, D> {
    fn volume(&self) -> N {
        // TODO: Is there a better way to handle this constraint?
        if D == 0 {
            panic!("Cannot calculate volume of bounds with D = 0")
        }

        self.min
            .iter()
            .zip(self.max.iter())
            .map(|(min, max)| *max - *min)
            .reduce(|acc, length| acc * length)
            .unwrap()
    }

    fn volume_increase_of_min_bounds(&self, other: &Self) -> N {
        min_bounds(self, other).volume() - self.volume()
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
