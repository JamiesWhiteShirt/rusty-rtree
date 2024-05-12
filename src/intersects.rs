use std::ops::Sub;

use crate::{bounds::SAABB, vector::SVec};

pub trait Intersects<T> {
    fn intersects(&self, rhs: &T) -> bool;
}

fn f64_min(lhs: f64, rhs: f64) -> f64 {
    if lhs < rhs {
        lhs
    } else {
        rhs
    }
}

fn f64_max(lhs: f64, rhs: f64) -> f64 {
    if lhs > rhs {
        lhs
    } else {
        rhs
    }
}

/// https://tavianator.com/2022/ray_box_boundary.html
///
/// NOTE: Although the source material does not mention it, it is not required
/// for the ray normal to have a magnitude of 1. This implementation substitutes
/// the normal for a delta value with unconstrained magnitude.
///
/// This is equivalent because in all comparisons, both operands are inversely
/// scaled with the magnitude of the delta.
pub(crate) fn ray_bounds_intersect<N, const D: usize>(
    b: &SAABB<N, D>,
    origin: &SVec<N, D>,
    delta: &SVec<N, D>,
) -> bool
where
    N: Clone + Sub<Output = N> + Into<f64>,
{
    let n_inv = 1.0 / (delta[0].clone()).into();
    let t1 = (b.min[0].clone() - origin[0].clone()).into() * n_inv;
    let t2 = (b.max[0].clone() - origin[0].clone()).into() * n_inv;

    let mut tmin = f64_min(t1, t2);
    let mut tmax = f64_max(t1, t2);

    for dim in 1..D {
        let n_inv = 1.0 / (delta[dim].clone()).into();
        let t1 = (b.min[dim].clone() - origin[dim].clone()).into() * n_inv;
        let t2 = (b.max[dim].clone() - origin[dim].clone()).into() * n_inv;

        tmin = f64_min(f64_max(t1, tmin), f64_max(t2, tmin));
        tmax = f64_max(f64_min(t1, tmax), f64_min(t2, tmax));
    }

    return tmin < tmax;
}

pub(crate) fn line_bounds_intersect<N, const D: usize>(
    b: &SAABB<N, D>,
    origin: &SVec<N, D>,
    delta: &SVec<N, D>,
) -> bool
where
    N: Clone + Sub<Output = N> + Into<f64>,
{
    let mut tmin: f64 = 0.0;
    let mut tmax: f64 = 1.0;

    for dim in 0..D {
        let n_inv = 1.0 / (delta[dim].clone()).into();
        let t1 = (b.min[dim].clone() - origin[dim].clone()).into() * n_inv;
        let t2 = (b.max[dim].clone() - origin[dim].clone()).into() * n_inv;

        tmin = f64_min(f64_max(t1, tmin), f64_max(t2, tmin));
        tmax = f64_max(f64_min(t1, tmax), f64_min(t2, tmax));
    }

    return tmin <= tmax;
}

#[cfg(test)]
mod test {
    use crate::{
        bounds::{AABB, SAABB},
        intersects::line_bounds_intersect,
        vector::SVec,
    };

    #[test]
    fn test_line_bounds_intersect() {
        let bounds: SAABB<i32, 2> = AABB {
            min: SVec([0, 0]),
            max: SVec([5, 5]),
        };
        let origin = SVec([6, 0]);
        let delta = SVec([-5, 5]);

        assert!(line_bounds_intersect(&bounds, &origin, &delta));
    }

    #[test]
    fn test_line_bounds_not_intersect() {
        let bounds: SAABB<i32, 2> = AABB {
            min: SVec([0, 0]),
            max: SVec([4, 4]),
        };
        let origin = SVec([-2, 3]);
        let delta = SVec([4, 4]);
        assert!(!line_bounds_intersect(&bounds, &origin, &delta));

        let origin = SVec([5, 2]);
        let delta = SVec([4, 0]);
        assert!(!line_bounds_intersect(&bounds, &origin, &delta));
    }
}
