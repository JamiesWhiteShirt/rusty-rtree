#![feature(test)]

mod bounds;
mod fc_vec;
mod filter;
mod intersects;
mod iter;
mod line;
mod node;
mod ray;
mod select;
mod sphere;
mod split;
mod util;
mod vector;

use bounds::Bounded;
use filter::SpatialFilter;
use iter::FilterIter;
use node::{Node, NodeContainer, NodeEntry, NodeOps, NodeRef, NodeRefMut};
use std::{fmt::Debug, ops::Sub};

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RTreeConfig {
    pub max_children: usize,
    pub min_children: usize,
}

/// R-tree spatial index with a map-like interface. It optimizes for
/// fast queries for objects that intersect a given space.
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
pub struct RTree<N, const D: usize, Key, Value> {
    config: RTreeConfig,
    height: usize,
    root: Node<N, D, Key, Value>,
}

impl<N, const D: usize, Key, Value> Debug for RTree<N, D, Key, Value>
where
    N: Debug,
    Key: Debug,
    Value: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ops = NodeOps::<N, D, Key, Value>::new_ops(self.config.max_children);
        f.debug_struct("RTree")
            .field("config", &self.config)
            .field("root", unsafe {
                &NodeRef::new(&ops, self.height, &self.root)
            })
            .finish()
    }
}

impl<N, const D: usize, Key, Value> Drop for RTree<N, D, Key, Value> {
    fn drop(&mut self) {
        let ops = NodeOps::<N, D, Key, Value>::new_ops(self.config.max_children);
        unsafe {
            ops.drop(&mut self.root, self.height);
        }
    }
}

impl<N, const D: usize, Key, Value> RTree<N, D, Key, Value>
where
    N: Ord + num_traits::Bounded,
    Key: Bounded<N, D>,
{
    pub fn iter<'a>(&'a self) -> iter::Iter<'a, N, D, Key, Value> {
        unsafe { iter::Iter::new(self.height, &self.root) }
    }

    pub fn filter_iter<'a, Filter: SpatialFilter<N, D, Key>>(
        &'a self,
        filter: Filter,
    ) -> FilterIter<'a, N, D, Key, Value, Filter> {
        unsafe { FilterIter::new(self.height, &self.root, filter) }
    }

    pub fn new(config: RTreeConfig) -> RTree<N, D, Key, Value> {
        let ops = NodeOps::<N, D, Key, Value>::new_ops(config.max_children);
        return RTree {
            height: 0,
            root: unsafe { ops.emtpy_leaf() },
            config,
        };
    }
}

impl<N, const D: usize, Key, Value> Clone for RTree<N, D, Key, Value>
where
    N: Clone,
    Key: Clone,
    Value: Clone,
{
    fn clone(&self) -> Self {
        let ops = NodeOps::<N, D, Key, Value>::new_ops(self.config.max_children);
        RTree {
            height: self.height,
            root: unsafe { ops.clone(&self.root, self.height) },
            config: self.config,
        }
    }
}

impl<N, const D: usize, Key, Value> RTree<N, D, Key, Value>
where
    N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
    Key: Bounded<N, D>,
{
    /// Inserts an entry into a node at the given level.
    unsafe fn insert_entry(&mut self, level: usize, entry: NodeEntry<N, D, Key, Value>) {
        let ops = NodeOps::<N, D, Key, Value>::new_ops(self.config.max_children);
        if let Some(sibling) = unsafe {
            ops.insert(
                &mut self.root,
                self.height,
                self.config.min_children,
                self.height - level,
                entry,
            )
        } {
            ops.branch(&mut self.root, self.height, sibling);
            self.height += 1;
        }
    }

    pub fn insert(&mut self, key: Key, value: Value) {
        unsafe {
            self.insert_entry(0, NodeEntry::Leaf((key, value)));
        }
    }

    pub fn insert_unique(&mut self, key: Key, value: Value) -> Option<Value>
    where
        Key: Eq,
    {
        unsafe {
            let ops = NodeOps::<N, D, Key, Value>::new_ops(self.config.max_children);
            let (prev_value, sibling) = ops.insert_unique(
                &mut self.root,
                self.height,
                self.config.min_children,
                key,
                value,
            );
            if let Some(sibling) = sibling {
                ops.branch(&mut self.root, self.height, sibling);
                self.height += 1;
            }
            prev_value
        }
    }

    pub fn get(&self, key: &Key) -> Option<&Value>
    where
        Key: Eq,
    {
        let ops = NodeOps::<N, D, Key, Value>::new_ops(self.config.max_children);
        let root_ref = unsafe { NodeRef::new(&ops, self.height, &self.root) };
        root_ref.get(key)
    }

    pub fn get_mut(&mut self, key: &Key) -> Option<&mut Value>
    where
        Key: Eq,
    {
        let ops = NodeOps::<N, D, Key, Value>::new_ops(self.config.max_children);
        let root_ref = unsafe { NodeRefMut::new(&ops, self.height, &mut self.root) };
        root_ref.get_mut(key)
    }

    pub fn remove(&mut self, key: &Key) -> Option<Value>
    where
        Key: Eq,
        Value: Eq,
    {
        let ops = NodeOps::<N, D, Key, Value>::new_ops(self.config.max_children);
        let height = self.height;
        let mut underfull_nodes: Box<[Option<Node<N, D, Key, Value>>]> =
            std::iter::repeat_with(|| None).take(height).collect();
        if let Some(value) = unsafe {
            ops.remove(
                &mut self.root,
                self.config.min_children,
                height,
                key,
                &mut underfull_nodes,
            )
        } {
            if unsafe { ops.try_unbranch(&mut self.root, self.height) } {
                self.height -= 1;
            }

            // reinsert entries at leaf level
            if height > 0 {
                if let Some(undefull_leaf) = underfull_nodes[0].take() {
                    let children = unsafe { ops.take_leaf_children(undefull_leaf) };
                    for leaf_entry in children.into_iter() {
                        unsafe {
                            self.insert_entry(0, NodeEntry::Leaf(leaf_entry));
                        }
                    }
                }
            }

            // reinsert entries at inner levels
            for level in 1..height {
                if let Some(children) = underfull_nodes[level].take() {
                    let children = unsafe { ops.take_inner_children(children) };
                    for node in children.into_iter() {
                        unsafe {
                            self.insert_entry(
                                level,
                                NodeEntry::Inner(NodeContainer::new(&ops, level - 1, node)),
                            );
                        }
                    }
                }
            }

            return Some(value);
        }
        return None;
    }

    pub fn len(&self) -> usize {
        let ops = NodeOps::<N, D, Key, Value>::new_ops(self.config.max_children);
        let root_ref = unsafe { NodeRef::new(&ops, self.height, &self.root) };
        root_ref.len()
    }

    fn debug_assert_bvh(&self)
    where
        N: Debug,
    {
        unsafe {
            self.root.debug_assert_bvh(self.height);
        }
    }

    fn debug_assert_eq(a: &Self, b: &Self)
    where
        N: Debug + Eq,
        Key: Debug + Eq,
        Value: Debug + Eq,
    {
        assert_eq!(a.config, b.config);
        assert_eq!(a.height, b.height);
        unsafe { Node::<N, D, Key, Value>::debug_assert_eq(&a.root, &b.root, a.height) }
    }

    fn debug_assert_min_children(&self)
    where
        N: Debug,
    {
        unsafe {
            self.root
                .debug_assert_min_children(self.height, self.config.min_children, true);
        }
    }
}

#[cfg(test)]
mod tests {
    use core::fmt;
    use std::error::Error;
    use std::fs::File;
    extern crate test;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use test::{black_box, Bencher};

    use noisy_float::types::n32;
    use noisy_float::types::N32;

    use crate::filter::BoundedIntersectionFilter;
    use crate::line::Line;
    use crate::sphere::Sphere;
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
        let mut tree = RTree::<i32, 2, Bounds<i32, 2>, i32>::new(RTreeConfig {
            max_children: 4,
            min_children: 2,
        });

        {
            let mut rng = StdRng::from_seed([
                0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD,
                0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF,
                0xDE, 0xAD, 0xBE, 0xEF,
            ]);
            for i in 0..1000 {
                let min = Vector([rng.gen_range(0..991), rng.gen_range(0..991)]);
                let max = min + Vector([rng.gen_range(1..11), rng.gen_range(1..11)]);
                tree.insert(Bounds { min, max }, i);
                tree.debug_assert_bvh();
            }
        }

        let len_before = tree.len();

        let key = Bounds {
            min: Vector([-1, -1]),
            max: Vector([-1, -1]),
        };
        // If this fails, this key was generated by chance
        assert_eq!(tree.get(&key), None);

        assert_eq!(tree.insert_unique(key, -1), None);
        assert_eq!(*tree.get(&key).unwrap(), -1);
        assert_eq!(tree.len(), len_before + 1);

        assert_eq!(tree.insert_unique(key, -2), Some(-1));
        assert_eq!(*tree.get(&key).unwrap(), -2);

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
            for i in 0..1000 {
                let min = Vector([rng.gen_range(0..991), rng.gen_range(0..991)]);
                let max = min + Vector([rng.gen_range(1..11), rng.gen_range(1..11)]);
                tree.insert(Bounds { min, max }, i);
                tree.debug_assert_bvh();
            }
        }

        let clone = tree.clone();
        RTree::<i32, 2, Bounds<i32, 2>, i32>::debug_assert_eq(&tree, &clone);
    }

    #[test]
    fn remove() {
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
            for i in 0..1000 {
                let min = Vector([rng.gen_range(0..991), rng.gen_range(0..991)]);
                let max = min + Vector([rng.gen_range(1..11), rng.gen_range(1..11)]);
                tree.insert(Bounds { min, max }, i);
                tree.debug_assert_bvh();
                tree.debug_assert_min_children();
            }
        }

        assert!(tree.len() == 1000);

        {
            let mut rng = StdRng::from_seed([
                0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD,
                0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF,
                0xDE, 0xAD, 0xBE, 0xEF,
            ]);
            for i in 0..1000 {
                let min = Vector([rng.gen_range(0..991), rng.gen_range(0..991)]);
                let max = min + Vector([rng.gen_range(1..11), rng.gen_range(1..11)]);
                assert_eq!(tree.remove(&Bounds { min, max }), Some(i));
                tree.debug_assert_bvh();
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
    fn insert_bench_4(bencher: &mut Bencher) {
        do_insert_bench(bencher, 4);
    }

    #[bench]
    fn insert_bench_8(bencher: &mut Bencher) {
        do_insert_bench(bencher, 8);
    }

    #[bench]
    fn insert_bench_16(bencher: &mut Bencher) {
        do_insert_bench(bencher, 16);
    }

    #[bench]
    fn insert_bench_32(bencher: &mut Bencher) {
        do_insert_bench(bencher, 32);
    }

    #[bench]
    fn insert_bench_64(bencher: &mut Bencher) {
        do_insert_bench(bencher, 64);
    }

    #[bench]
    fn insert_bench_128(bencher: &mut Bencher) {
        do_insert_bench(bencher, 128);
    }

    #[bench]
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
            for entry in tree.filter_iter(BoundedIntersectionFilter::new(Bounds { min, max })) {
                black_box(entry);
            }
        });
    }

    #[bench]
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

        stars.debug_assert_bvh();
        stars.debug_assert_min_children();

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

        for (pos, info) in stars.filter_iter(BoundedIntersectionFilter::new(space)) {
            if info.proper.len() > 0 {
                println!("{}", info.proper);
            }

            star_lines.insert(Line(sol_pos, *pos), ())
        }

        star_lines.debug_assert_bvh();
        star_lines.debug_assert_min_children();

        Ok(())
    }
}
