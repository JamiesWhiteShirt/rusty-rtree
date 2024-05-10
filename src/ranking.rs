use std::{
    cmp,
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

use array_init::from_iter;
use itertools::izip;

use crate::{bounds::AABB, vector::Vector};

/// A trait for defining a weak ordering of R-tree keys by a metric. In addition
/// to providing a metric for keys, a ranking can also provide a lower bound
/// for the metric for [`Bounds`] containing a set of keys.
pub trait Ranking<B, Key>
where
    Key: ?Sized,
{
    /// The metric used to order keys. This must implement [`Ord`].
    type Metric: Ord;

    /// Returns a lower bound for the metric for keys contained by the given
    /// bounds. That is, for all keys `k` contained by `bounds`,
    /// `rank_key(k) >= bounds_min(bounds)`.
    ///
    /// The lower bound does not need to be tight, but it should be as tight as
    /// possible to improve search performance. When searching in an R-tree,
    /// nodes are searched in order of increasing lower bound, and entries can
    /// only be yielded if their metric is no larger than the lower bounds of
    /// all nodes that are yet to be searched. The tighter the lower bound, the
    /// fewer nodes need to be searched.
    ///
    /// It must also hold that for all bounds `b1` and `b2`, if `b1` contains
    /// `b2`, then `bounds_min(b1) <= bounds_min(b2)`. In other words, removing
    /// keys from a set of keys cannot decrease the lower bound for the metric.
    fn bounds_min(&self, bounds: &B) -> Self::Metric;

    /// Returns the metric for the given key.
    fn rank_key(&self, key: &Key) -> Self::Metric;
}

pub trait PointDistance<N, const D: usize> {
    fn dist_sq(&self, point: &Vector<N, D>) -> N;
}

impl<N, const D: usize> PointDistance<N, D> for Vector<N, D>
where
    N: Clone + Sub<Output = N> + Mul<Output = N> + Add<Output = N>,
{
    fn dist_sq(&self, point: &Vector<N, D>) -> N {
        (self.clone() - point.clone()).sq_mag()
    }
}

/// Orders keys by euclidean distance to a given point. Requires that the key
/// type implements [`PointDistance`]. The metric used by this is the squared
/// euclidean distance as `N`. `N` must support multiplication, addition and
/// subtraction, and must have a sufficient range to represent the squared
/// distance between the given point and the furthest point in the R-tree's
/// bounds. Failure to support the required range may cause numeric overflow and
/// incorrect results.
///
/// In general, `N` should likely be some totally ordered floating point number
/// to avoid overflow, though this is not always required. For R-trees with
/// integer coordinates, `N` can be an integer type provided it has sufficient
/// range.
#[derive(Clone, Copy, Debug)]
pub struct EuclideanDistanceRanking<N, const D: usize, Key>
where
    Key: ?Sized + PointDistance<N, D>,
    N: Ord + Clone + Sub<Output = N> + Mul<Output = N> + Add<Output = N>,
{
    point: Vector<N, D>,
    _phantom: PhantomData<Key>,
}

impl<N, const D: usize, Key> EuclideanDistanceRanking<N, D, Key>
where
    Key: ?Sized + PointDistance<N, D>,
    N: Ord + Clone + Sub<Output = N> + Mul<Output = N> + Add<Output = N>,
{
    pub fn new(point: Vector<N, D>) -> Self {
        Self {
            point,
            _phantom: PhantomData,
        }
    }
}

impl<N, const D: usize, Key, T> Ranking<AABB<T, D>, Key> for EuclideanDistanceRanking<N, D, Key>
where
    Key: ?Sized + PointDistance<N, D>,
    N: Ord + Clone + Sub<Output = N> + Mul<Output = N> + Add<Output = N>,
    T: Clone + Into<N>,
{
    type Metric = N;

    fn bounds_min(&self, bounds: &AABB<T, D>) -> Self::Metric {
        let closest_point: Vector<N, D> = Vector(
            from_iter(
                izip!(&bounds.min.0, &bounds.max.0, &self.point.0).map(|(min, max, pt)| {
                    cmp::max(min.clone().into(), cmp::min(max.clone().into(), pt.clone()))
                }),
            )
            .unwrap(),
        )
        .into_map(Into::into);

        closest_point.dist_sq(&self.point)
    }

    fn rank_key(&self, key: &Key) -> Self::Metric {
        key.dist_sq(&self.point)
    }
}
