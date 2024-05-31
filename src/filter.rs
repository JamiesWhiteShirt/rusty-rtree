use crate::{bounds::Bounded, contains::Contains, intersects::Intersects};

/// A spatial search filter for R-tree keys. In addition to testing individual
/// keys, spatial filters can also be used to prune the search space by testing
/// bounds containing sets of keys.
pub trait SpatialFilter<B, Key>
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
    fn test_bounds(&self, bounds: &B) -> bool;

    /// Returns `true` if the given key matches the filter.
    fn test_key(&self, key: &Key) -> bool;
}

/// Matches keys that intersect the given space. To be used as a spatial filter,
/// the space must implement [`Intersects`] with both the bounds type and the
/// key type.
pub struct IntersectsFilter<S> {
    space: S,
}

impl<S> IntersectsFilter<S> {
    pub fn new(space: S) -> Self {
        Self { space }
    }
}

impl<B, S, Key> SpatialFilter<B, Key> for IntersectsFilter<S>
where
    S: Intersects<B> + Intersects<Key>,
{
    fn test_bounds(&self, bounds: &B) -> bool {
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
pub struct BoundedIntersectsFilter<B, S> {
    bounds: B,
    space: S,
}

impl<B, S> BoundedIntersectsFilter<B, S>
where
    S: Bounded<B>,
{
    /// Creates a new filter that matches keys that intersect the given space
    /// using the space's bounds for intersection testing with other bounds.
    pub fn new_bounded(space: S) -> Self {
        Self {
            bounds: space.bounds(),
            space,
        }
    }
}

impl<B, S, Key> SpatialFilter<B, Key> for BoundedIntersectsFilter<B, S>
where
    B: Intersects<B>,
    S: Intersects<Key>,
{
    fn test_bounds(&self, bounds: &B) -> bool {
        self.bounds.intersects(bounds)
    }

    fn test_key(&self, key: &Key) -> bool {
        self.space.intersects(key)
    }
}

/// Matches keys that are contained by the given space. To be used as a spatial
/// filter, the space must implement [`Intersects`] with the bounds type and
/// [`Contains`] with the key type.
pub struct ContainsFilter<S> {
    space: S,
}

impl<S> ContainsFilter<S> {
    pub fn new(space: S) -> Self {
        Self { space }
    }
}

impl<B, S, Key> SpatialFilter<B, Key> for ContainsFilter<S>
where
    B: Intersects<B>,
    S: Intersects<B> + Contains<Key>,
{
    fn test_bounds(&self, bounds: &B) -> bool {
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
pub struct BoundedContainsFilter<B, S> {
    bounds: B,
    space: S,
}

impl<B, S> BoundedContainsFilter<B, S>
where
    S: Bounded<B>,
{
    pub fn new_bounded(space: S) -> Self {
        Self {
            bounds: space.bounds(),
            space,
        }
    }
}

impl<B, S, Key> SpatialFilter<B, Key> for BoundedContainsFilter<B, S>
where
    B: Intersects<B>,
    S: Contains<Key>,
{
    fn test_bounds(&self, bounds: &B) -> bool {
        self.bounds.intersects(bounds)
    }

    fn test_key(&self, key: &Key) -> bool {
        self.space.contains(key)
    }
}

/// Matches all keys.
pub struct NoFilter;

impl<B, Key> SpatialFilter<B, Key> for NoFilter {
    fn test_bounds(&self, _: &B) -> bool {
        true
    }

    fn test_key(&self, _: &Key) -> bool {
        true
    }
}

/// A join filter for R-tree keys. In addition to testing pair of keys, join
/// filters can also be used to prune the search space by testing pairs of
/// bounds containing sets of keys.
pub trait JoinFilter<BL, BR, KeyL, KeyR>
where
    KeyL: ?Sized,
    KeyR: ?Sized,
{
    /// Returns `true` if a pair of keys contained by the given pair of bounds
    /// could match the filter.
    ///
    /// If a pair of keys contained by the pair of bounds could match the
    /// filter, this must return `true`, otherwise pairs of keys matching the
    /// filter would be pruned from the search. If no pair of keys contained by
    /// the pair of bounds could match the filter, this should return `false`,
    /// otherwise the filter will needlessly be applied to pairs of keys or
    /// pairs of subsets of keys that cannot match the filter.
    ///
    /// When joining two R-trees, this is used to determine whether the children
    /// of a pair of nodes should be joined or pruned. If this returns `false`,
    /// the pair will be pruned and their children will not be joined.
    fn test_bounds(&self, left: &BL, right: &BR) -> bool;

    /// Returns `true` if the given pair of keys matches the filter.
    fn test_key(&self, left: &KeyL, right: &KeyR) -> bool;
}

/// Matches keys matching a join with the given key and bounds. Used by joining
/// iterators to filter for right keys that match a left key.
pub struct JoiningFilter<'a, B, Key, Filter>
where
    Key: ?Sized,
{
    bounds: B,
    key: &'a Key,
    filter: Filter,
}

impl<'a, B, Key, Filter> JoiningFilter<'a, B, Key, Filter>
where
    Key: ?Sized + Bounded<B>,
{
    pub fn new(key: &'a Key, filter: Filter) -> Self {
        Self {
            bounds: key.bounds(),
            key,
            filter,
        }
    }
}

impl<'a, BL, BR, Key0, Key1, Filter> SpatialFilter<BR, Key1> for JoiningFilter<'a, BL, Key0, Filter>
where
    Key0: ?Sized,
    Key1: ?Sized,
    Filter: JoinFilter<BL, BR, Key0, Key1>,
{
    fn test_bounds(&self, bounds: &BR) -> bool {
        self.filter.test_bounds(&self.bounds, bounds)
    }

    fn test_key(&self, key: &Key1) -> bool {
        self.filter.test_key(self.key, key)
    }
}
