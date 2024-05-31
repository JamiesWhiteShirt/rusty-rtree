use std::{
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

use crate::{
    bounds::AABB,
    vector::{SVec, Vector},
};

/// A trait for defining a weak ordering of R-tree keys by a metric. In addition
/// to providing a metric for keys, a ranking can also provide a lower bound
/// for the metric for bounds containing a set of keys.
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

pub trait PointDistance<V, N> {
    fn dist_sq(&self, point: &V) -> N;
}

impl<N, const D: usize> PointDistance<SVec<N, D>, N> for SVec<N, D>
where
    N: Clone + Sub<Output = N> + Mul<Output = N> + Add<Output = N>,
{
    fn dist_sq(&self, point: &SVec<N, D>) -> N {
        (self.clone() - point.clone()).sq_mag()
    }
}

impl<V, N> PointDistance<V, N> for AABB<V>
where
    V: Vector + PointDistance<V, N>,
{
    fn dist_sq(&self, point: &V) -> N {
        let closest_point = point
            .componentwise_max(&self.min)
            .componentwise_min(&self.max);

        closest_point.dist_sq(point)
    }
}

/// Orders keys by euclidean distance to a given point. Requires that the key
/// type implements [`PointDistance`]. The metric used by this is the squared
/// euclidean distance as `M`. `M` must have a sufficient range to represent the
/// squared distance between the given point and the furthest point in the
/// R-tree's bounds. Failure to support the required range may cause numeric
/// overflow and incorrect results.
///
/// In general, `M` should likely be some totally ordered floating point number
/// to avoid overflow, though this is not always required. For R-trees with
/// integer coordinates, `N` can be an integer type provided it has sufficient
/// range.
#[derive(Clone, Copy, Debug)]
pub struct EuclideanDistanceRanking<V, M> {
    point: V,
    _phantom: PhantomData<M>,
}

impl<V, M> EuclideanDistanceRanking<V, M> {
    pub fn new(point: V) -> Self {
        Self {
            point,
            _phantom: PhantomData,
        }
    }
}

impl<B, V, Key, M> Ranking<B, Key> for EuclideanDistanceRanking<V, M>
where
    Key: ?Sized + PointDistance<V, M>,
    B: PointDistance<V, M>,
    M: Ord,
{
    type Metric = M;

    fn bounds_min(&self, bounds: &B) -> Self::Metric {
        bounds.dist_sq(&self.point)
    }

    fn rank_key(&self, key: &Key) -> Self::Metric {
        key.dist_sq(&self.point)
    }
}
