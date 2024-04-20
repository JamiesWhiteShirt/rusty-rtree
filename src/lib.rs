#![feature(test)]

pub mod bounds;
pub mod contains;
mod fc_vec;
pub mod filter;
pub mod geom;
pub mod intersects;
mod iter;
mod iter_stack;
mod join;
mod node;
pub mod ranking;
mod select;
mod split;
mod util;
pub mod vector;

use bounds::Bounded;
use filter::SpatialFilter;
use iter::FilterIter;
use node::{Node, NodeOps, NodeRef, NodeRefMut, RootNodeRefMut};
use ranking::Ranking;
use std::{borrow::Borrow, fmt::Debug, ops::Sub};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RTreeConfig {
    pub max_children: usize,
    pub min_children: usize,
}

/// R-tree spatial index with a map-like interface. It optimizes for spatial
/// queries, such as finding objects that intersect a given space, or finding
/// the nearest object(s) to a given point.
///
/// The R-tree operates on objects that are bounded in `D` dimensions measured
/// in scalars of type `N`. Each entry in the R-tree is a key-value pair, where
/// the `Key` is a bounded object, while the `Value` may be used to store
/// additional data associated with the object.
///
/// Unlike a traditional map, a single key may be associated with multiple
/// values.
///
/// The R-tree is parameterized by a `RTreeConfig` that specifies the maximum
/// and minimum number of children per node. This configuration will affect
/// the performance of the R-tree in different operations. The optimal
/// configuration will depend on the use case.
///
/// # Safety
///
/// Here be dragons. The R-tree is implemented using unsafe code.
pub struct RTree<N, const D: usize, Key, Value>
where
    N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
    Key: Bounded<N, D> + Eq,
{
    config: RTreeConfig,
    height: usize,
    root: Node<N, D, Key, Value>,
}

impl<N, const D: usize, Key, Value> Debug for RTree<N, D, Key, Value>
where
    N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64> + Debug,
    Key: Bounded<N, D> + Eq + Debug,
    Value: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RTree")
            .field("config", &self.config)
            .field("root", &self.node_ref())
            .finish()
    }
}

impl<N, const D: usize, Key, Value> Drop for RTree<N, D, Key, Value>
where
    N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
    Key: Bounded<N, D> + Eq,
{
    fn drop(&mut self) {
        let ops = self.ops();
        unsafe {
            ops.wrap_ref_mut(&mut self.root, self.height).drop();
        }
    }
}

impl<N, const D: usize, Key, Value> Clone for RTree<N, D, Key, Value>
where
    N: Ord + num_traits::Bounded + Sub<Output = N> + Into<f64> + Clone,
    Key: Bounded<N, D> + Eq + Clone,
    Value: Clone,
{
    fn clone(&self) -> Self {
        let ops = self.ops();
        let root = unsafe { ops.wrap_ref(&self.root, self.height) }.clone();
        RTree {
            height: self.height,
            root: root.unwrap(),
            config: self.config,
        }
    }
}

impl<N, const D: usize, Key, Value> RTree<N, D, Key, Value>
where
    N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
    Key: Bounded<N, D> + Eq,
{
    fn ops(&self) -> NodeOps {
        NodeOps::new_ops(self.config.min_children, self.config.max_children)
    }

    fn root_ref_mut(&mut self) -> RootNodeRefMut<N, D, Key, Value> {
        let ops = self.ops();
        unsafe { ops.wrap_root_ref_mut(&mut self.root, &mut self.height) }
    }

    fn node_ref_mut(&mut self) -> NodeRefMut<N, D, Key, Value> {
        let ops = self.ops();
        unsafe { ops.wrap_ref_mut(&mut self.root, self.height) }
    }

    fn node_ref(&self) -> NodeRef<N, D, Key, Value> {
        let ops = self.ops();
        unsafe { ops.wrap_ref(&self.root, self.height) }
    }

    /// Returns a new empty R-tree.
    pub fn new(config: RTreeConfig) -> RTree<N, D, Key, Value> {
        let ops = NodeOps::new_ops(config.min_children, config.max_children);
        return RTree {
            height: 0,
            root: ops.empty_leaf().unwrap(),
            config,
        };
    }

    /// Returns an iterator over all entries in the R-tree.
    pub fn iter<'a>(&'a self) -> iter::Iter<'a, N, D, Key, Value> {
        unsafe { iter::Iter::new(self.height, &self.root) }
    }

    /// Returns a mutable iterator over all entries in the R-tree.
    pub fn iter_mut<'a>(&'a mut self) -> iter::IterMut<'a, N, D, Key, Value> {
        unsafe { iter::IterMut::new(self.height, &mut self.root) }
    }

    /// Returns an iterator over all entries in the R-tree using a spatial
    /// filter.
    pub fn filter_iter<'a, Q, Filter>(
        &'a self,
        filter: Filter,
    ) -> FilterIter<'a, N, D, Key, Value, Q, Filter>
    where
        Key: Borrow<Q>,
        Q: ?Sized,
        Filter: SpatialFilter<N, D, Q>,
    {
        unsafe { FilterIter::new(self.height, &self.root, filter) }
    }

    /// Returns a mutable iterator over all entries in the R-tree using a
    /// spatial filter.
    pub fn filter_iter_mut<'a, Q, Filter: SpatialFilter<N, D, Q>>(
        &'a mut self,
        filter: Filter,
    ) -> iter::FilterIterMut<'a, N, D, Key, Value, Q, Filter>
    where
        Key: Borrow<Q>,
        Q: ?Sized,
        Filter: SpatialFilter<N, D, Q>,
    {
        unsafe { iter::FilterIterMut::new(self.height, &mut self.root, filter) }
    }

    /// Returns an entry with a minimal key according to the given [`Ranking`].
    /// If there are multiple entries with minimal keys, either because the keys
    /// have the same rank or multiple entries with the same key exist, the
    /// first entry is returned.
    pub fn min_by<R>(&self, ranking: R) -> Option<(&Key, &Value)>
    where
        R: Ranking<N, D, Key>,
    {
        self.iter_asc_by(ranking).next()
    }

    /// Returns a mutable entry with a minimal key according to the given
    /// [`Ranking`]. If there are multiple entries with minimal keys, either
    /// because the keys have the same rank or multiple entries with the same
    /// key exist, the first entry is returned.
    pub fn min_by_mut<R>(&mut self, ranking: R) -> Option<(&Key, &mut Value)>
    where
        R: Ranking<N, D, Key>,
    {
        self.iter_asc_by_mut(ranking).next()
    }

    /// Returns an entry with a minimal key according to the given [`Ranking`],
    /// filtering out entries whose key rank metric is None. If there are
    /// multiple entries with minimal keys, either because the keys have the
    /// same rank or multiple entries with the same key exist, the first entry
    /// is returned.
    ///
    /// The [`Ranking`] invariants still apply using Option<S> order. If
    /// `bounds_min(bounds)` is None, it implies that for all keys `k` contained
    /// by `bounds`, `rank_key(k)` is None.
    pub fn filter_min_by<R, S>(&self, ranking: R) -> Option<(&Key, &Value)>
    where
        R: Ranking<N, D, Key, Metric = Option<S>>,
        S: Ord,
    {
        self.filter_iter_asc_by(ranking).next()
    }

    /// Returns a mutable entry with a minimal key according to the given
    /// [`Ranking`], filtering out entries whose key rank metric is None. If
    /// there are multiple entries with minimal keys, either because the keys
    /// have the same rank or multiple entries with the same key exist, the
    /// first entry is returned.
    ///
    /// The [`Ranking`] invariants still apply using Option<S> order. If
    /// `bounds_min(bounds)` is None, it implies that for all keys `k` contained
    /// by `bounds`, `rank_key(k)` is None.
    pub fn filter_min_by_mut<R, S>(&mut self, ranking: R) -> Option<(&Key, &mut Value)>
    where
        R: Ranking<N, D, Key, Metric = Option<S>>,
        S: Ord,
    {
        self.filter_iter_asc_by_mut(ranking).next()
    }

    /// Returns an iterator over the entries in the R-tree in _ascending key
    /// order_ according to the given [`Ranking`].
    ///
    /// To only get the entry with the least key, use [`RTree::min_by`].
    pub fn iter_asc_by<R>(&self, ranking: R) -> iter::SortedIter<N, D, Key, Value, R>
    where
        R: Ranking<N, D, Key>,
    {
        unsafe { iter::SortedIter::new(self.height, &self.root, ranking) }
    }

    /// Returns a mutable iterator over the entries in the R-tree in _ascending
    /// key order_ according to the given [`Ranking`].
    ///
    /// To only get the entry with the least key, use [`RTree::min_by_mut`].
    pub fn iter_asc_by_mut<R>(&mut self, ranking: R) -> iter::SortedIterMut<N, D, Key, Value, R>
    where
        R: Ranking<N, D, Key>,
    {
        unsafe { iter::SortedIterMut::new(self.height, &mut self.root, ranking) }
    }

    /// Returns an iterator over the entries in the R-tree in _ascending key
    /// order_ according to the given [`Ranking`], filtering out entries whose
    /// key rank metric is None.
    ///
    /// The [`Ranking`] invariants still apply using Option<S> order. If
    /// `bounds_min(bounds)` is None, it implies that for all keys `k` contained
    /// by `bounds`, `rank_key(k)` is None.
    ///
    /// To only get the entry with the least key with the same filtering, use
    /// [`RTree::filter_min_by`].
    pub fn filter_iter_asc_by<R, S>(
        &self,
        ranking: R,
    ) -> iter::FilterSortedIter<N, D, Key, Value, R, S>
    where
        R: Ranking<N, D, Key, Metric = Option<S>>,
        S: Ord,
    {
        unsafe { iter::FilterSortedIter::new(self.height, &self.root, ranking) }
    }

    /// Returns a mutable iterator over the entries in the R-tree in _ascending
    /// key order_ according to the given [`Ranking`], filtering out entries
    /// whose key rank metric is None.
    ///
    /// The [`Ranking`] invariants still apply using Option<S> order. If
    /// `bounds_min(bounds)` is None, it implies that for all keys `k` contained
    /// by `bounds`, `rank_key(k)` is None.
    ///
    /// To only get the entry with the least key with the same filtering, use
    /// [`RTree::filter_min_by_mut`].
    pub fn filter_iter_asc_by_mut<R, S>(
        &mut self,
        ranking: R,
    ) -> iter::FilterSortedIterMut<N, D, Key, Value, R, S>
    where
        R: Ranking<N, D, Key, Metric = Option<S>>,
        S: Ord,
    {
        unsafe { iter::FilterSortedIterMut::new(self.height, &mut self.root, ranking) }
    }

    pub fn join<'a, 'b, N1, const D1: usize, Key1, Value1, F>(
        &'a self,
        other: &'b RTree<N1, D1, Key1, Value1>,
        filter: F,
    ) -> join::JoinIter<'a, 'b, N, N1, D, D1, Key, Key1, Value, Value1, F>
    where
        N1: Ord + num_traits::Bounded + Clone + Sub<Output = N1> + Into<f64>,
        Key1: Bounded<N1, D1> + Eq,
        F: join::JoinFilter<N, N1, D, D1, Key, Key1>,
    {
        unsafe { join::JoinIter::new(filter, &self.root, self.height, &other.root, other.height) }
    }

    // Inserts a new key-value pair into the R-tree, ignoring any existing
    // entries with the same key.
    pub fn insert(&mut self, key: Key, value: Value) {
        let ops = self.ops();
        let mut root = unsafe { ops.wrap_root_ref_mut(&mut self.root, &mut self.height) };
        root.insert(key, value)
    }

    /// Inserts a new key-value pair into the R-tree, or replaces the value of
    /// an existing entry with the same key, returning the previous value. If
    /// multiple entries with the same key exist, the value of the first entry
    /// will be replaced.
    pub fn insert_unique(&mut self, key: Key, value: Value) -> Option<Value> {
        let ops = self.ops();
        let mut root = unsafe { ops.wrap_root_ref_mut(&mut self.root, &mut self.height) };
        root.insert_unique(key, value)
    }

    /// Returns a reference to the value associated with the given key, or `None`
    /// if no such value exists. If multiple entries with the same key exist, a
    /// reference to the the value of the first entry is returned.
    ///
    /// Borrowing is done using the `Borrow` trait, so the key can be of a
    /// different type than the key type of the R-tree. The Bounded trait must
    /// be equivalent for borrowed and owned keys, like Eq, Ord and Hash.
    pub fn get<Q>(&self, key: &Q) -> Option<&Value>
    where
        Key: Borrow<Q>,
        Q: Eq + Bounded<N, D> + ?Sized,
    {
        self.node_ref().get(key)
    }

    /// Returns a mutable reference to the value associated with the given key,
    /// or `None` if no such value exists. If multiple entries with the same key
    /// exist, a reference to the the value of the first entry is returned.
    ///
    /// Borrowing is done using the `Borrow` trait, so the key can be of a
    /// different type than the key type of the R-tree. The Bounded trait must
    /// be equivalent for borrowed and owned keys, like Eq, Ord and Hash.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut Value>
    where
        Key: Borrow<Q>,
        Q: Eq + Bounded<N, D> + ?Sized,
    {
        self.node_ref_mut().into_get_mut(key)
    }

    /// Removes the entry with the given key, returning the value of the entry
    /// if it existed. If multiple entries with the same key exist, the first
    /// entry is removed.
    ///
    /// Borrowing is done using the `Borrow` trait, so the key can be of a
    /// different type than the key type of the R-tree. The Bounded trait must
    /// be equivalent for borrowed and owned keys, like Eq, Ord and Hash.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<Value>
    where
        Key: Borrow<Q>,
        Q: Eq + Bounded<N, D> + ?Sized,
    {
        self.root_ref_mut().remove(key)
    }

    /// Returns the number of entries in the R-tree.
    pub fn len(&self) -> usize {
        let ops = self.ops();
        let root = unsafe { ops.wrap_ref(&self.root, self.height) };
        root.len()
    }

    fn _debug_assert_bvh(&self)
    where
        N: Debug,
    {
        self.node_ref()._debug_assert_bvh();
    }

    fn _debug_assert_eq(a: &Self, b: &Self)
    where
        N: Debug + Eq,
        Key: Debug + Eq,
        Value: Debug + Eq,
    {
        assert_eq!(a.config, b.config);
        assert_eq!(a.height, b.height);
        a.node_ref()._debug_assert_eq(&b.node_ref());
    }

    fn _debug_assert_min_children(&self)
    where
        N: Debug,
    {
        self.node_ref()._debug_assert_min_children(true);
    }
}

impl<'a, N, const D: usize, Key, Value> IntoIterator for &'a RTree<N, D, Key, Value>
where
    N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
    Key: Bounded<N, D> + Eq,
{
    type Item = (&'a Key, &'a Value);
    type IntoIter = iter::Iter<'a, N, D, Key, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, N, const D: usize, Key, Value> IntoIterator for &'a mut RTree<N, D, Key, Value>
where
    N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
    Key: Bounded<N, D> + Eq,
{
    type Item = (&'a Key, &'a mut Value);
    type IntoIter = iter::IterMut<'a, N, D, Key, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use core::fmt;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::error::Error;
    use std::fs::File;
    extern crate test;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use test::{black_box, Bencher};

    use noisy_float::types::n32;
    use noisy_float::types::N32;

    use crate::bounds::Bounded;
    use crate::filter::BoundedIntersectsFilter;
    use crate::geom::line::Line;
    use crate::geom::sphere::Sphere;
    use crate::intersects::Intersects;
    use crate::join::JoinFilter;
    use crate::ranking::EuclideanDistanceRanking;
    use crate::ranking::PointDistance;
    use crate::vector::Vector;
    use crate::RTreeConfig;

    use super::bounds::Bounds;
    use super::RTree;

    #[test]
    fn insert() {
        let mut tree = RTree::<i32, 2, Bounds<i32, 2>, ()>::new(RTreeConfig {
            max_children: 4,
            min_children: 2,
        });

        tree.insert(
            Bounds {
                min: Vector([0, 0]),
                max: Vector([1, 1]),
            },
            (),
        );

        tree.insert(
            Bounds {
                min: Vector([2, 0]),
                max: Vector([3, 1]),
            },
            (),
        );

        tree.insert(
            Bounds {
                min: Vector([0, 2]),
                max: Vector([1, 3]),
            },
            (),
        );

        tree.insert(
            Bounds {
                min: Vector([2, 2]),
                max: Vector([3, 3]),
            },
            (),
        );

        tree.insert(
            Bounds {
                min: Vector([0, 2]),
                max: Vector([0, 2]),
            },
            (),
        );

        tree.insert(
            Bounds {
                min: Vector([1, 3]),
                max: Vector([1, 3]),
            },
            (),
        );
    }

    #[test]
    fn get() {
        let mut tree = RTree::<i32, 2, Bounds<i32, 2>, bool>::new(RTreeConfig {
            max_children: 4,
            min_children: 2,
        });

        let key = Bounds {
            min: Vector([0, 0]),
            max: Vector([1, 1]),
        };
        tree.insert(key, true);

        assert!(tree.get(&key).unwrap());
    }

    #[test]
    fn get_mut() {
        let mut tree = RTree::<i32, 2, Bounds<i32, 2>, bool>::new(RTreeConfig {
            max_children: 4,
            min_children: 2,
        });

        let key = Bounds {
            min: Vector([0, 0]),
            max: Vector([1, 1]),
        };
        tree.insert(key, false);

        *tree.get_mut(&key).unwrap() = true;

        assert!(tree.get(&key).unwrap());
    }

    #[test]
    fn insert_unique() {
        let mut tree = RTree::<i32, 2, Bounds<i32, 2>, u16>::new(RTreeConfig {
            max_children: 4,
            min_children: 2,
        });

        {
            let mut rng = StdRng::from_seed([
                0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD,
                0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF,
                0xDE, 0xAD, 0xBE, 0xEF,
            ]);
            for i in 0..50 {
                let min = Vector([rng.gen_range(0..991), rng.gen_range(0..991)]);
                let max = min + Vector([rng.gen_range(1..11), rng.gen_range(1..11)]);
                tree.insert(Bounds { min, max }, i);
                tree._debug_assert_bvh();
                assert_eq!(tree.len(), usize::from(i + 1))
            }
        }

        let len_before = tree.len();

        let key = Bounds {
            min: Vector([-1, -1]),
            max: Vector([-1, -1]),
        };
        // If this fails, this key was generated by chance
        assert_eq!(tree.get(&key), None);

        assert_eq!(tree.insert_unique(key, 1001), None);
        assert_eq!(*tree.get(&key).unwrap(), 1001);
        assert_eq!(tree.len(), len_before + 1);

        assert_eq!(tree.insert_unique(key, 1002), Some(1001));
        assert_eq!(*tree.get(&key).unwrap(), 1002);

        assert_eq!(tree.len(), len_before + 1);
    }

    #[test]
    fn clone() {
        let mut tree: RTree<i32, 2, Bounds<i32, 2>, i32> =
            RTree::<i32, 2, Bounds<i32, 2>, i32>::new(RTreeConfig {
                max_children: 4,
                min_children: 2,
            });

        {
            let mut rng = StdRng::from_seed([
                0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD,
                0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF,
                0xDE, 0xAD, 0xBE, 0xEF,
            ]);
            for i in 0..50 {
                let min = Vector([rng.gen_range(0..991), rng.gen_range(0..991)]);
                let max = min + Vector([rng.gen_range(1..11), rng.gen_range(1..11)]);
                tree.insert(Bounds { min, max }, i);
                tree._debug_assert_bvh();
            }
        }

        let clone = tree.clone();
        RTree::_debug_assert_eq(&tree, &clone);
    }

    #[test]
    fn remove() {
        let mut tree: RTree<i32, 2, Bounds<i32, 2>, i32> = RTree::new(RTreeConfig {
            max_children: 4,
            min_children: 2,
        });

        {
            let mut rng = StdRng::from_seed([
                0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD,
                0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF,
                0xDE, 0xAD, 0xBE, 0xEF,
            ]);
            for i in 0..50 {
                let min = Vector([rng.gen_range(0..991), rng.gen_range(0..991)]);
                let max = min + Vector([rng.gen_range(1..11), rng.gen_range(1..11)]);
                tree.insert(Bounds { min, max }, i);
                tree._debug_assert_bvh();
                tree._debug_assert_min_children();
            }
        }

        {
            let mut rng = StdRng::from_seed([
                0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD,
                0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF,
                0xDE, 0xAD, 0xBE, 0xEF,
            ]);
            for i in 0..50 {
                let min = Vector([rng.gen_range(0..991), rng.gen_range(0..991)]);
                let max = min + Vector([rng.gen_range(1..11), rng.gen_range(1..11)]);
                assert_eq!(tree.remove(&Bounds { min, max }), Some(i));
                tree._debug_assert_bvh();
            }
        }

        assert!(tree.len() == 0);
    }

    fn do_insert_bench(bencher: &mut Bencher, max_children: usize) {
        let min_children = max_children / 4;
        let mut rng = StdRng::from_seed([
            0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD,
            0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF,
            0xDE, 0xAD, 0xBE, 0xEF,
        ]);

        let mut tree = RTree::<i32, 2, Bounds<i32, 2>, i32>::new(RTreeConfig {
            max_children,
            min_children,
        });
        for i in 0..10000 {
            let min = Vector([rng.gen_range(0..991), rng.gen_range(0..991)]);
            let max = min + Vector([rng.gen_range(1..11), rng.gen_range(1..11)]);
            tree.insert(Bounds { min, max }, i);
        }

        bencher.iter(|| {
            let min = Vector([rng.gen_range(0..991), rng.gen_range(0..991)]);
            let max = min + Vector([10, 10]);
            tree.insert(Bounds { min, max }, 0)
        });
    }

    #[bench]
    #[cfg_attr(miri, ignore)]
    fn insert_bench_4(bencher: &mut Bencher) {
        do_insert_bench(bencher, 4);
    }

    #[bench]
    #[cfg_attr(miri, ignore)]
    fn insert_bench_8(bencher: &mut Bencher) {
        do_insert_bench(bencher, 8);
    }

    #[bench]
    #[cfg_attr(miri, ignore)]
    fn insert_bench_16(bencher: &mut Bencher) {
        do_insert_bench(bencher, 16);
    }

    #[bench]
    #[cfg_attr(miri, ignore)]
    fn insert_bench_32(bencher: &mut Bencher) {
        do_insert_bench(bencher, 32);
    }

    #[bench]
    #[cfg_attr(miri, ignore)]
    fn insert_bench_64(bencher: &mut Bencher) {
        do_insert_bench(bencher, 64);
    }

    #[bench]
    #[cfg_attr(miri, ignore)]
    fn insert_bench_128(bencher: &mut Bencher) {
        do_insert_bench(bencher, 128);
    }

    #[bench]
    #[cfg_attr(miri, ignore)]
    fn insert_bench_256(bencher: &mut Bencher) {
        do_insert_bench(bencher, 256);
    }

    fn do_query_bench(bencher: &mut Bencher, max_children: usize) {
        let mut rng = StdRng::from_seed([
            0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD,
            0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF,
            0xDE, 0xAD, 0xBE, 0xEF,
        ]);
        let mut tree = RTree::<i32, 2, Bounds<i32, 2>, i32>::new(RTreeConfig {
            max_children,
            min_children: max_children / 2,
        });
        for i in 0..10000 {
            let min = Vector([rng.gen_range(0..991), rng.gen_range(0..991)]);
            let max = min + Vector([rng.gen_range(1..11), rng.gen_range(1..11)]);
            tree.insert(Bounds { min, max }, i);
        }

        bencher.iter(|| {
            let min = Vector([rng.gen_range(0..991), rng.gen_range(0..991)]);
            let max = min + Vector([10, 10]);
            for entry in tree.filter_iter(BoundedIntersectsFilter::new_bounded(Bounds { min, max }))
            {
                black_box(entry);
            }
        });
    }

    #[bench]
    #[cfg_attr(miri, ignore)]
    fn query_bench_64(bencher: &mut Bencher) {
        do_query_bench(bencher, 64);
    }

    struct StarInfo {
        id: u32,
        proper: String,
    }

    fn record_to_star(
        record: csv::StringRecord,
    ) -> Result<(Vector<N32, 3>, StarInfo), Box<dyn Error>> {
        let id: u32 = record[0].parse()?;
        let proper: String = record[6].parse()?;
        let x: N32 = n32(record[17].parse()?);
        let y: N32 = n32(record[18].parse()?);
        let z: N32 = n32(record[19].parse()?);

        Ok((Vector([x, y, z]), StarInfo { id, proper }))
    }

    #[derive(Debug, Clone)]
    struct SolNotFoundError;

    impl Error for SolNotFoundError {}

    impl fmt::Display for SolNotFoundError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Sol not found!")
        }
    }

    #[test]
    fn cosmic() -> Result<(), Box<dyn Error>> {
        let mut stars = RTree::<N32, 3, Vector<N32, 3>, StarInfo>::new(RTreeConfig {
            min_children: 4,
            max_children: 32,
        });

        let file = File::open("./hygdata_v3.csv");
        let mut rdr = csv::Reader::from_reader(file?);
        let mut iter = rdr.records();

        let (sol_pos, sol_info) = record_to_star(iter.next().ok_or(Box::new(SolNotFoundError))??)?;
        stars.insert(sol_pos, sol_info);

        for result in iter {
            let (pos, info) = record_to_star(result?)?;
            stars.insert(pos, info);
        }

        stars._debug_assert_bvh();
        stars._debug_assert_min_children();

        let space: Sphere<N32, 3> = Sphere {
            center: sol_pos,
            radius: n32(100.0),
        };
        let bounds: Bounds<N32, 3> = Bounds {
            min: sol_pos.into_map(|coord| coord - 100.0),
            max: sol_pos.into_map(|coord| coord + 100.0),
        };

        let mut star_lines = RTree::<N32, 3, Line<N32, 3>, ()>::new(RTreeConfig {
            min_children: 4,
            max_children: 32,
        });

        for (pos, info) in stars.filter_iter(BoundedIntersectsFilter::new_bounded(space)) {
            if info.proper.len() > 0 {
                println!("{}", info.proper);
            }

            star_lines.insert(Line(sol_pos, *pos), ())
        }

        star_lines._debug_assert_bvh();
        star_lines._debug_assert_min_children();

        Ok(())
    }

    #[test]
    fn test_drop() {
        struct CountedUnit<'a>(&'a RefCell<i32>);

        impl<'a> CountedUnit<'a> {
            fn new(counter: &'a RefCell<i32>) -> Self {
                *counter.borrow_mut() += 1;
                Self(counter)
            }
        }

        impl<'a> Drop for CountedUnit<'a> {
            fn drop(&mut self) {
                *self.0.borrow_mut() -= 1;
            }
        }

        struct Key<'a> {
            i: u32,
            _unit: CountedUnit<'a>,
        }

        impl<'a> Bounded<u32, 2> for Key<'a> {
            fn bounds(&self) -> Bounds<u32, 2> {
                Bounds {
                    min: Vector([self.i, self.i]),
                    max: Vector([self.i + 1, self.i + 1]),
                }
            }
        }

        impl<'a> PartialEq for Key<'a> {
            fn eq(&self, other: &Self) -> bool {
                self.i == other.i
            }
        }

        impl<'a> Eq for Key<'a> {}

        let value_count = RefCell::new(0);
        let key_count = RefCell::new(0);
        let mut tree = RTree::<u32, 2, Key, CountedUnit>::new(RTreeConfig {
            max_children: 4,
            min_children: 2,
        });

        for i in 0..50 {
            tree.insert(
                Key {
                    i,
                    _unit: CountedUnit::new(&key_count),
                },
                CountedUnit::new(&value_count),
            );
            assert_eq!(key_count.borrow().clone(), (i + 1).try_into().unwrap());
            assert_eq!(value_count.borrow().clone(), (i + 1).try_into().unwrap());
        }

        drop(tree);
        assert_eq!(key_count.borrow().clone(), 0);
        assert_eq!(value_count.borrow().clone(), 0);
    }

    #[test]
    fn sorted_iter_asc() {
        let mut tree = RTree::<i32, 2, Vector<i32, 2>, ()>::new(RTreeConfig {
            max_children: 4,
            min_children: 2,
        });

        for x in 0..10 {
            for y in 0..10 {
                tree.insert(Vector([x * 10, y * 10]), ());
            }
        }

        let mut visited = [[false; 10]; 10];
        fn mark_visited(visited: &mut [[bool; 10]; 10], key: &Vector<i32, 2>) {
            let x = key[0] as usize / 10;
            let y = key[1] as usize / 10;
            assert!(!visited[x][y]);
            visited[x][y] = true;
        }

        let center = Vector([9, 9]);
        let ranking = EuclideanDistanceRanking::<i32, 2, Vector<i32, 2>>::new(center);

        let mut iter = tree.iter_asc_by(ranking);
        let mut last_dist = {
            let (key, _) = iter.next().unwrap();
            mark_visited(&mut visited, key);
            center.dist_sq(key)
        };

        for (key, _) in iter {
            let dist = center.dist_sq(key);
            assert!(dist >= last_dist);
            last_dist = dist;
            mark_visited(&mut visited, key);
        }

        for x in 0..10 {
            for y in 0..10 {
                assert!(visited[x][y]);
            }
        }
    }

    #[test]
    fn join() {
        let mut tree0 = RTree::<i32, 2, Vector<i32, 2>, i32>::new(RTreeConfig {
            max_children: 4,
            min_children: 2,
        });
        let mut tree1 = RTree::<i32, 2, Vector<i32, 2>, i32>::new(RTreeConfig {
            max_children: 4,
            min_children: 2,
        });

        for x in 0..8 {
            for y in 0..8 {
                tree0.insert(Vector([x, y]), x * y);
            }
        }
        for x in 4..20 {
            for y in 4..20 {
                tree1.insert(Vector([x, y]), x * y);
            }
        }

        struct EqFilter;

        impl<N, const D: usize, Key> JoinFilter<N, N, D, D, Key, Key> for EqFilter
        where
            N: Ord,
            Key: Eq,
        {
            fn test_bounds(&self, bounds0: &Bounds<N, D>, bounds1: &Bounds<N, D>) -> bool {
                bounds0.intersects(bounds1)
            }

            fn test_key(&self, key0: &Key, key1: &Key) -> bool {
                key0 == key1
            }
        }

        let mut visited = HashSet::new();
        for ((key0, value0), (key1, value1)) in tree0.join(&tree1, EqFilter) {
            assert!(
                visited.insert(((*key0, *value0), (*key1, *value1))),
                "Duplicate entry yielded"
            );
        }
        let mut expected = HashSet::new();
        for x in 4..8 {
            for y in 4..8 {
                expected.insert(((Vector([x, y]), x * y), (Vector([x, y]), x * y)));
            }
        }
        assert_eq!(visited, expected);
    }
}
