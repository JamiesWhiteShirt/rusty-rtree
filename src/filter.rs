use crate::{
    bounds::{Bounded, Bounds},
    contains::Contains,
    intersects::Intersects,
};

/// A spatial search filter for R-tree keys. In addition to testing individual
/// keys, spatial filters can also be used to prune the search space by testing
/// bounds containing sets of keys.
pub trait SpatialFilter<N, const D: usize, Key>
where
    Key: ?Sized,
{
    /// Returns `true` if a key contained by the given bounds could match the
    /// filter.
    ///
    /// If a key contained by the bounds could match the filter, this must
    /// return `true`, otherwise keys matching the filter would be pruned from
    /// the search. If no key contained by the bounds could match the filter,
    /// this should return `false`, otherwise the filter will needlessly be
    /// applied to keys or subsets of keys that cannot match the filter.
    ///
    /// When searching in an R-tree, this is used to determine whether the
    /// contents of an R-tree node should be tested against the filter or
    /// pruned. If this returns `false`, the node will be pruned and its
    /// children will not be tested.
    fn test_bounds(&self, bounds: &Bounds<N, D>) -> bool;

    /// Returns `true` if the given key matches the filter.
    fn test_key(&self, key: &Key) -> bool;
}

/// Matches keys that intersect the given space. To be used as a spatial filter,
/// the space must implement [`Intersects`] with both [`Bounds`] and with the
/// key type.
pub struct IntersectsFilter<S> {
    space: S,
}

impl<S> IntersectsFilter<S> {
    pub fn new(space: S) -> Self {
        Self { space }
    }
}

impl<N, const D: usize, S, Key> SpatialFilter<N, D, Key> for IntersectsFilter<S>
where
    N: Ord,
    S: Intersects<Bounds<N, D>> + Intersects<Key>,
{
    fn test_bounds(&self, bounds: &Bounds<N, D>) -> bool {
        self.space.intersects(bounds)
    }

    fn test_key(&self, key: &Key) -> bool {
        self.space.intersects(key)
    }
}

/// Matches keys that intersect the given space and bounds. To be used as a
/// spatial filter, the space must implement [`Intersects`] with the key type.
///
/// Unlike [`IntersectsFilter`], BoundedInsersectsFilter uses bounds to test for
/// intersection with other bounds. This limits the accuracy of the filter's
/// bounds test, but may be more efficient in cases where testing for
/// intersection between the space and bounds is expensive.
pub struct BoundedIntersectsFilter<N, const D: usize, S>
where
    N: Ord,
{
    bounds: Bounds<N, D>,
    space: S,
}

impl<N, const D: usize, S> BoundedIntersectsFilter<N, D, S>
where
    N: Ord,
{
    /// Creates a new filter that matches keys that intersect the given space
    /// using the space's bounds for intersection testing with other bounds.
    pub fn new_bounded(space: S) -> Self
    where
        S: Bounded<N, D>,
    {
        Self {
            bounds: space.bounds(),
            space,
        }
    }
}

impl<N, const D: usize, S, Key> SpatialFilter<N, D, Key> for BoundedIntersectsFilter<N, D, S>
where
    N: Ord,
    S: Intersects<Key>,
{
    fn test_bounds(&self, bounds: &Bounds<N, D>) -> bool {
        self.bounds.intersects(bounds)
    }

    fn test_key(&self, key: &Key) -> bool {
        self.space.intersects(key)
    }
}

/// Matches keys that are contained by the given space. To be used as a spatial
/// filter, the space must implement [`Intersects`] with [`Bounds`] and
/// [`Contains`] with the key type.
pub struct ContainsFilter<S> {
    space: S,
}

impl<S> ContainsFilter<S> {
    pub fn new(space: S) -> Self {
        Self { space }
    }
}

impl<N, const D: usize, S, Key> SpatialFilter<N, D, Key> for ContainsFilter<S>
where
    N: Ord,
    S: Intersects<Bounds<N, D>> + Contains<Key>,
{
    fn test_bounds(&self, bounds: &Bounds<N, D>) -> bool {
        self.space.intersects(bounds)
    }

    fn test_key(&self, key: &Key) -> bool {
        self.space.contains(key)
    }
}

/// Matches keys that are contained by the given space and bounds. To be used as
/// a spatial filter, the space must implement [`Contains`] with the key type.
///
/// Unlike [`ContainsFilter`], BoundedContainsFilter uses bounds to test for
/// intersection with other bounds. This limits the accuracy of the filter's
/// bounds test, but may be more efficient in cases where testing for
/// intersection between the space and bounds is expensive.
pub struct BoundedContainsFilter<N, const D: usize, S>
where
    N: Ord,
    S: Bounded<N, D>,
{
    bounds: Bounds<N, D>,
    space: S,
}

impl<N, const D: usize, S> BoundedContainsFilter<N, D, S>
where
    N: Ord,
    S: Bounded<N, D>,
{
    pub fn new_bounded(space: S) -> Self {
        Self {
            bounds: space.bounds(),
            space,
        }
    }
}

impl<N, const D: usize, S, Key> SpatialFilter<N, D, Key> for BoundedContainsFilter<N, D, S>
where
    N: Ord,
    S: Bounded<N, D> + Contains<Key>,
{
    fn test_bounds(&self, bounds: &Bounds<N, D>) -> bool {
        self.bounds.intersects(bounds)
    }

    fn test_key(&self, key: &Key) -> bool {
        self.space.contains(key)
    }
}

/// Matches all keys.
pub struct NoFilter;

impl<N, const D: usize, Key> SpatialFilter<N, D, Key> for NoFilter {
    fn test_bounds(&self, _: &Bounds<N, D>) -> bool {
        true
    }

    fn test_key(&self, _: &Key) -> bool {
        true
    }
}
