#![feature(ptr_metadata, layout_for_ptr)]

pub mod bounds;
pub mod contains;
mod fc_vec;
pub mod filter;
pub mod geom;
pub mod intersects;
mod iter;
mod iter_stack;
mod join;
mod left_join;
mod node;
pub mod ranking;
mod rc_mut;
mod select;
mod split;
mod util;
pub mod vector;

use bounds::{Bounded, Bounds, Volume};
use contains::Contains;
use filter::{JoinFilter, SpatialFilter};
use iter::{FilterIter, TreeIter, TreeIterMut};
use node::{Alloc, NodeContainer};
use ranking::Ranking;
use select::MinimalVolumeIncreaseSelector;
use split::QuadraticSplitter;
use std::{borrow::Borrow, fmt::Debug};

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
#[derive(Clone, Debug)]
pub struct RTree<B, Key, Value> {
    root: NodeContainer<B, Key, Value>,
}

impl<B, Key, Value> RTree<B, Key, Value> {
    /// Returns an iterator over all entries in the R-tree.
    pub fn iter<'a>(&'a self) -> iter::Iter<Box<TreeIter<'a, B, Key, Value>>> {
        iter::Iter::new(self.root.borrow())
    }

    /// Returns a mutable iterator over all entries in the R-tree.
    pub fn iter_mut<'a>(&'a mut self) -> iter::IterMut<Box<TreeIterMut<'a, B, Key, Value>>> {
        iter::IterMut::new(self.root.borrow_mut())
    }

    /// Returns a new empty R-tree.
    pub fn new(config: RTreeConfig) -> RTree<B, Key, Value>
    where
        B: Bounds,
    {
        let ops = Alloc::new_alloc(config.min_children, config.max_children);
        return RTree {
            root: ops.new_leaf(),
        };
    }

    /// Returns an iterator over all entries in the R-tree using a spatial
    /// filter.
    pub fn filter_iter<'a, Q, Filter>(
        &'a self,
        filter: Filter,
    ) -> FilterIter<Box<TreeIter<'a, B, Key, Value>>, Q, Filter>
    where
        Key: Borrow<Q>,
        Q: ?Sized,
        Filter: SpatialFilter<B, Q>,
    {
        FilterIter::new(self.root.borrow(), filter)
    }

    /// Returns a mutable iterator over all entries in the R-tree using a
    /// spatial filter.
    pub fn filter_iter_mut<'a, Q, Filter: SpatialFilter<B, Q>>(
        &'a mut self,
        filter: Filter,
    ) -> iter::FilterIterMut<Box<TreeIterMut<'a, B, Key, Value>>, Q, Filter>
    where
        Key: Borrow<Q>,
        Q: ?Sized,
        Filter: SpatialFilter<B, Q>,
    {
        iter::FilterIterMut::new(self.root.borrow_mut(), filter)
    }

    /// Returns an entry with a minimal key according to the given [`Ranking`].
    /// If there are multiple entries with minimal keys, either because the keys
    /// have the same rank or multiple entries with the same key exist, the
    /// first entry is returned.
    pub fn min_by<R>(&self, ranking: R) -> Option<(&Key, &Value)>
    where
        R: Ranking<B, Key>,
    {
        self.iter_asc_by(ranking).next()
    }

    /// Returns a mutable entry with a minimal key according to the given
    /// [`Ranking`]. If there are multiple entries with minimal keys, either
    /// because the keys have the same rank or multiple entries with the same
    /// key exist, the first entry is returned.
    pub fn min_by_mut<R>(&mut self, ranking: R) -> Option<(&Key, &mut Value)>
    where
        R: Ranking<B, Key>,
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
        R: Ranking<B, Key, Metric = Option<S>>,
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
        R: Ranking<B, Key, Metric = Option<S>>,
        S: Ord,
    {
        self.filter_iter_asc_by_mut(ranking).next()
    }

    /// Returns an iterator over the entries in the R-tree in _ascending key
    /// order_ according to the given [`Ranking`].
    ///
    /// To only get the entry with the least key, use [`RTree::min_by`].
    pub fn iter_asc_by<R>(&self, ranking: R) -> iter::SortedIter<B, Key, Value, R>
    where
        R: Ranking<B, Key>,
    {
        iter::SortedIter::new(self.root.borrow(), ranking)
    }

    /// Returns a mutable iterator over the entries in the R-tree in _ascending
    /// key order_ according to the given [`Ranking`].
    ///
    /// To only get the entry with the least key, use [`RTree::min_by_mut`].
    pub fn iter_asc_by_mut<R>(&mut self, ranking: R) -> iter::SortedIterMut<B, Key, Value, R>
    where
        R: Ranking<B, Key>,
    {
        iter::SortedIterMut::new(self.root.borrow_mut(), ranking)
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
    ) -> iter::FilterSortedIter<B, Key, Value, R, S>
    where
        R: Ranking<B, Key, Metric = Option<S>>,
        S: Ord,
    {
        iter::FilterSortedIter::new(self.root.borrow(), ranking)
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
    ) -> iter::FilterSortedIterMut<B, Key, Value, R, S>
    where
        R: Ranking<B, Key, Metric = Option<S>>,
        S: Ord,
    {
        iter::FilterSortedIterMut::new(self.root.borrow_mut(), ranking)
    }

    pub fn join<'a, 'b, B1, Key1, Value1, Q0, Q1, F>(
        &'a self,
        other: &'b RTree<B1, Key1, Value1>,
        filter: F,
    ) -> join::JoinIter<'a, 'b, B, B1, Key, Key1, Value, Value1, Q0, Q1, F>
    where
        Key: Borrow<Q0>,
        Key1: Borrow<Q1>,
        Q0: ?Sized,
        Q1: ?Sized,
        Key1: Bounded<B1> + Eq,
        F: JoinFilter<B, B1, Q0, Q1>,
    {
        join::JoinIter::new(self.root.borrow(), other.root.borrow(), filter)
    }

    pub fn left_join<'a, 'b, B1, Key1, Value1, Q0, Q1, F>(
        &'a self,
        other: &'b RTree<B1, Key1, Value1>,
        filter: F,
    ) -> left_join::LeftJoinIter<'a, 'b, B, B1, Key, Key1, Value, Value1, Q0, Q1, F>
    where
        Key: Borrow<Q0>,
        Key1: Borrow<Q1>,
        Q0: ?Sized,
        Q1: ?Sized,
        Key1: Bounded<B1> + Eq,
        F: JoinFilter<B, B1, Q0, Q1>,
    {
        left_join::LeftJoinIter::new(self.root.borrow(), other.root.borrow(), filter)
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
        B: Contains<B>,
        Key: Borrow<Q>,
        Q: Eq + Bounded<B> + ?Sized,
    {
        self.root.borrow().get(key)
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
        B: Contains<B>,
        Key: Borrow<Q>,
        Q: Eq + Bounded<B> + ?Sized,
    {
        self.root.borrow_mut().into_get_mut(key)
    }

    /// Returns the number of entries in the R-tree.
    pub fn len(&self) -> usize {
        self.root.borrow().len()
    }

    fn _debug_assert_bvh(&self)
    where
        B: Debug + Bounds + Eq,
        Key: Bounded<B>,
    {
        self.root.borrow()._debug_assert_bvh();
    }

    fn _debug_assert_eq(a: &Self, b: &Self)
    where
        B: Debug + Eq,
        Key: Debug + Eq,
        Value: Debug + Eq,
    {
        a.root._debug_assert_eq(&b.root)
    }

    fn _debug_assert_min_children(&self)
    where
        B: Debug,
    {
        self.root._debug_assert_min_children();
    }
}

impl<B, Key, Value> RTree<B, Key, Value>
where
    B: Bounds + Volume + Clone,
    Key: Bounded<B> + Eq,
{
    // Inserts a new key-value pair into the R-tree, ignoring any existing
    // entries with the same key.
    pub fn insert(&mut self, key: Key, value: Value) {
        self.root.insert(
            &mut MinimalVolumeIncreaseSelector,
            &mut QuadraticSplitter,
            key,
            value,
        )
    }

    /// Inserts a new key-value pair into the R-tree, or replaces the value of
    /// an existing entry with the same key, returning the previous value. If
    /// multiple entries with the same key exist, the value of the first entry
    /// will be replaced.
    pub fn insert_unique(&mut self, key: Key, value: Value) -> Option<Value> {
        self.root.insert_unique(
            &mut MinimalVolumeIncreaseSelector,
            &mut QuadraticSplitter,
            key,
            value,
        )
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
        Q: Eq + Bounded<B> + ?Sized,
    {
        self.root.remove(
            &mut MinimalVolumeIncreaseSelector,
            &mut QuadraticSplitter,
            key,
        )
    }
}

impl<'a, B, Key, Value> IntoIterator for &'a RTree<B, Key, Value>
where
    Key: Bounded<B>,
{
    type Item = (&'a Key, &'a Value);
    type IntoIter = iter::Iter<Box<TreeIter<'a, B, Key, Value>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, B, Key, Value> IntoIterator for &'a mut RTree<B, Key, Value>
where
    Key: Bounded<B>,
{
    type Item = (&'a Key, &'a mut Value);
    type IntoIter = iter::IterMut<Box<TreeIterMut<'a, B, Key, Value>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use core::fmt;
    use noisy_float::types::n64;
    use noisy_float::types::N64;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::error::Error;
    use std::fs::File;
    use std::ops::Add;
    use std::ops::Mul;
    use std::ops::Sub;

    use noisy_float::types::n32;
    use noisy_float::types::N32;

    use crate::bounds::Bounded;
    use crate::filter::BoundedIntersectsFilter;
    use crate::filter::JoinFilter;
    use crate::geom::line::Line;
    use crate::geom::sphere::Sphere;
    use crate::ranking::EuclideanDistanceRanking;
    use crate::ranking::PointDistance;
    use crate::vector::SVec;
    use crate::RTreeConfig;

    use super::bounds::SAABB;
    use super::RTree;

    #[test]
    fn insert() {
        let mut tree = RTree::<SAABB<i32, 2>, SAABB<i32, 2>, ()>::new(RTreeConfig {
            max_children: 4,
            min_children: 2,
        });

        tree.insert(
            SAABB {
                min: SVec([0, 0]),
                max: SVec([1, 1]),
            },
            (),
        );

        tree.insert(
            SAABB {
                min: SVec([2, 0]),
                max: SVec([3, 1]),
            },
            (),
        );

        tree.insert(
            SAABB {
                min: SVec([0, 2]),
                max: SVec([1, 3]),
            },
            (),
        );

        tree.insert(
            SAABB {
                min: SVec([2, 2]),
                max: SVec([3, 3]),
            },
            (),
        );

        tree.insert(
            SAABB {
                min: SVec([0, 2]),
                max: SVec([0, 2]),
            },
            (),
        );

        tree.insert(
            SAABB {
                min: SVec([1, 3]),
                max: SVec([1, 3]),
            },
            (),
        );
    }

    #[test]
    fn get() {
        let mut tree = RTree::<SAABB<i32, 2>, SAABB<i32, 2>, bool>::new(RTreeConfig {
            max_children: 4,
            min_children: 2,
        });

        let key = SAABB {
            min: SVec([0, 0]),
            max: SVec([1, 1]),
        };
        tree.insert(key, true);

        assert!(tree.get(&key).unwrap());
    }

    #[test]
    fn get_mut() {
        let mut tree = RTree::<SAABB<i32, 2>, SAABB<i32, 2>, bool>::new(RTreeConfig {
            max_children: 4,
            min_children: 2,
        });

        let key = SAABB {
            min: SVec([0, 0]),
            max: SVec([1, 1]),
        };
        tree.insert(key, false);

        *tree.get_mut(&key).unwrap() = true;

        assert!(tree.get(&key).unwrap());
    }

    #[test]
    fn insert_unique() {
        let mut tree = RTree::<SAABB<i32, 2>, SAABB<i32, 2>, u16>::new(RTreeConfig {
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
                let min = SVec([rng.gen_range(0..991), rng.gen_range(0..991)]);
                let max = min + SVec([rng.gen_range(1..11), rng.gen_range(1..11)]);
                tree.insert(SAABB { min, max }, i);
                tree._debug_assert_bvh();
                assert_eq!(tree.len(), usize::from(i + 1))
            }
        }

        let len_before = tree.len();

        let key = SAABB {
            min: SVec([-1, -1]),
            max: SVec([-1, -1]),
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
        let mut tree: RTree<SAABB<i32, 2>, SAABB<i32, 2>, i32> =
            RTree::<SAABB<i32, 2>, SAABB<i32, 2>, i32>::new(RTreeConfig {
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
                let min = SVec([rng.gen_range(0..991), rng.gen_range(0..991)]);
                let max = min + SVec([rng.gen_range(1..11), rng.gen_range(1..11)]);
                tree.insert(SAABB { min, max }, i);
                tree._debug_assert_bvh();
            }
        }

        let clone = tree.clone();
        RTree::_debug_assert_eq(&tree, &clone);
    }

    #[test]
    fn remove() {
        let mut tree: RTree<SAABB<i32, 2>, SAABB<i32, 2>, i32> = RTree::new(RTreeConfig {
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
                let min = SVec([rng.gen_range(0..991), rng.gen_range(0..991)]);
                let max = min + SVec([rng.gen_range(1..11), rng.gen_range(1..11)]);
                tree.insert(SAABB { min, max }, i);
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
                let min = SVec([rng.gen_range(0..991), rng.gen_range(0..991)]);
                let max = min + SVec([rng.gen_range(1..11), rng.gen_range(1..11)]);
                assert_eq!(tree.remove(&SAABB { min, max }), Some(i));
                tree._debug_assert_bvh();
            }
        }

        assert!(tree.len() == 0);
    }

    struct StarInfo {
        id: u32,
        proper: String,
    }

    fn record_to_star(
        record: csv::StringRecord,
    ) -> Result<(SVec<N32, 3>, StarInfo), Box<dyn Error>> {
        let id: u32 = record[0].parse()?;
        let proper: String = record[6].parse()?;
        let x: N32 = n32(record[17].parse()?);
        let y: N32 = n32(record[18].parse()?);
        let z: N32 = n32(record[19].parse()?);

        Ok((SVec([x, y, z]), StarInfo { id, proper }))
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
    #[cfg_attr(miri, ignore)]
    fn cosmic() -> Result<(), Box<dyn Error>> {
        let mut stars = RTree::<SAABB<N32, 3>, SVec<N32, 3>, StarInfo>::new(RTreeConfig {
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
        let bounds: SAABB<N32, 3> = SAABB {
            min: sol_pos.into_map(|coord| coord - 100.0),
            max: sol_pos.into_map(|coord| coord + 100.0),
        };

        let mut star_lines = RTree::<SAABB<N32, 3>, Line<N32, 3>, ()>::new(RTreeConfig {
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

        impl<'a> Bounded<SAABB<u32, 2>> for Key<'a> {
            fn bounds(&self) -> SAABB<u32, 2> {
                SAABB {
                    min: SVec([self.i, self.i]),
                    max: SVec([self.i + 1, self.i + 1]),
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
        let mut tree = RTree::<SAABB<u32, 2>, Key, CountedUnit>::new(RTreeConfig {
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
        let mut tree = RTree::<SAABB<i32, 2>, SVec<i32, 2>, ()>::new(RTreeConfig {
            max_children: 4,
            min_children: 2,
        });

        for x in 0..10 {
            for y in 0..10 {
                tree.insert(SVec([x * 10, y * 10]), ());
            }
        }

        let mut visited = [[false; 10]; 10];
        fn mark_visited(visited: &mut [[bool; 10]; 10], key: &SVec<i32, 2>) {
            let x = key[0] as usize / 10;
            let y = key[1] as usize / 10;
            assert!(!visited[x][y]);
            visited[x][y] = true;
        }

        let center = SVec([9, 9]);
        let ranking = EuclideanDistanceRanking::<SVec<i32, 2>, i32>::new(center);

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
        let mut rng = StdRng::from_seed([
            0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD,
            0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF,
            0xDE, 0xAD, 0xBE, 0xEF,
        ]);
        let tree0 = {
            let mut tree = RTree::<SAABB<i32, 2>, SVec<i32, 2>, usize>::new(RTreeConfig {
                max_children: 4,
                min_children: 2,
            });
            for i in 0..50usize {
                tree.insert(SVec([rng.gen_range(0..100), rng.gen_range(0..100)]), i);
            }
            tree
        };
        let tree1 = {
            let mut tree = RTree::<SAABB<i32, 2>, SVec<i32, 2>, usize>::new(RTreeConfig {
                max_children: 4,
                min_children: 2,
            });
            for i in 0..50usize {
                tree.insert(SVec([rng.gen_range(0..100), rng.gen_range(0..100)]), i);
            }
            tree
        };

        #[derive(Clone)]
        struct DistanceFilter(N64);

        impl<N, const D: usize> JoinFilter<SAABB<N, D>, SAABB<N, D>, SVec<N, D>, SVec<N, D>>
            for DistanceFilter
        where
            N: Ord + Clone + Add<Output = N> + Sub<Output = N> + Mul<Output = N> + Into<f64>,
        {
            fn test_bounds(&self, bounds0: &SAABB<N, D>, bounds1: &SAABB<N, D>) -> bool {
                bounds0.sq_dist_to(bounds1) < self.0 * self.0
            }

            fn test_key(&self, key0: &SVec<N, D>, key1: &SVec<N, D>) -> bool {
                n64((key0.clone() - key1.clone()).sq_mag().into()) < self.0 * self.0
            }
        }

        let filter = DistanceFilter(n64(5.0));

        let mut visited = HashSet::new();
        for ((key0, value0), r_iter) in tree0.left_join(&tree1, filter.clone()) {
            for (key1, value1) in r_iter {
                assert!(
                    visited.insert(((key0, value0), (key1, value1))),
                    "{:?} was yielded twice",
                    ((key0, value0), (key1, value1)),
                );
            }
        }
        let mut expected = HashSet::new();
        for (key0, value0) in tree0.iter() {
            for (key1, value1) in tree1.iter() {
                if filter.test_key(key0, key1) {
                    expected.insert(((key0, value0), (key1, value1)));
                }
            }
        }
        assert_eq!(visited, expected);
    }
}
