#![feature(test)]

mod bounds;
mod filter;
mod intersects;
mod iter;
mod line;
mod ray;
mod select;
mod sphere;
mod split;
mod vector;

use bounds::{min_bounds, min_bounds_all, Bounded, Bounds};
use filter::SpatialFilter;
use intersects::Intersects;
use iter::FilterIter;
use std::{
    mem::{ManuallyDrop, MaybeUninit},
    ops::Sub,
};

fn main() {
    println!("Hello, world!");
}

pub(crate) struct NodeRef<N, const D: usize, Key, Value> {
    pub(crate) bounds: Bounds<N, D>,
    pub(crate) node: Node<N, D, Key, Value>,
}

impl<N, const D: usize, Key, Value> Bounded<N, D> for NodeRef<N, D, Key, Value>
where
    N: Clone,
    Key: Bounded<N, D>,
{
    fn bounds(&self) -> Bounds<N, D> {
        self.bounds.clone()
    }
}

enum NodeEntry<N, const D: usize, Key, Value> {
    Inner(NodeRef<N, D, Key, Value>),
    Leaf((Key, Value)),
}

impl<N, const D: usize, Key, Value> Bounded<N, D> for NodeEntry<N, D, Key, Value>
where
    N: Clone,
    Key: Bounded<N, D>,
{
    fn bounds(&self) -> Bounds<N, D> {
        match self {
            NodeEntry::Inner(node_ref) => node_ref.bounds(),
            NodeEntry::Leaf((key, _)) => key.bounds(),
        }
    }
}

pub(crate) union Node<N, const D: usize, Key, Value> {
    inner: ManuallyDrop<Vec<NodeRef<N, D, Key, Value>>>,
    leaf: ManuallyDrop<Vec<(Key, Value)>>,
}

impl<N, const D: usize, Key, Value> Node<N, D, Key, Value> {
    unsafe fn drop(&mut self, level: usize) {
        if level > 0 {
            for child in &mut *self.inner {
                child.node.drop(level - 1);
            }
            ManuallyDrop::drop(&mut self.inner);
        } else {
            ManuallyDrop::drop(&mut self.leaf);
        }
    }

    unsafe fn len(&self, level: usize) -> usize {
        if level > 0 {
            self.inner.len()
        } else {
            self.leaf.len()
        }
    }

    unsafe fn compute_bounds(&self, level: usize) -> Option<Bounds<N, D>>
    where
        N: Ord + Clone,
        Key: Bounded<N, D>,
    {
        if level > 0 {
            min_bounds_all(self.inner.iter().map(|child| child.bounds()))
        } else {
            min_bounds_all(self.leaf.iter().map(|(key, _)| key.bounds()))
        }
    }

    unsafe fn remove(
        &mut self,
        min_children: usize,
        level: usize,
        key: &Key,
        value: &Value,
        reinsert_nodes: &mut [Option<Node<N, D, Key, Value>>],
    ) -> bool
    where
        N: Ord + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D> + Eq,
        Value: Eq,
    {
        if level > 0 {
            let children = &mut *self.inner;
            let mut i = children.len();
            while i > 0 {
                i -= 1;
                if children[i].bounds.intersects(&key.bounds())
                    && children[i]
                        .node
                        .remove(min_children, level - 1, key, value, reinsert_nodes)
                {
                    if children[i].node.len(level - 1) < min_children {
                        let removed_child = children.swap_remove(i);
                        reinsert_nodes[level - 1] = Some(removed_child.node);
                    } else {
                        children[i].recompute_bounds(level - 1);
                    }
                    return true;
                }
            }
            return false;
        } else {
            let children = &mut *self.leaf;
            let index = children.iter().position(|(k, v)| k == key && v == value);
            if let Some(i) = index {
                children.remove(i);
                return true;
            }
            return false;
        }
    }
}

impl<N, const D: usize, Key, Value> Bounded<N, D> for (Key, Value)
where
    N: Ord,
    Key: Bounded<N, D>,
{
    fn bounds(&self) -> Bounds<N, D> {
        self.0.bounds()
    }
}

impl<N, const D: usize, Key, Value> NodeRef<N, D, Key, Value>
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
    Key: Bounded<N, D>,
{
    /// Safety:
    /// - `depth` must be less than the height of the tree.
    /// - `entry` must be `NodeEntry::Inner` if nodes at the target `depth` are inner nodes, or `NodeEntry::Leaf` if they are leaf nodes.
    unsafe fn insert_entry(
        &mut self,
        max_children: usize,
        depth: usize,
        entry: NodeEntry<N, D, Key, Value>,
    ) -> Option<NodeRef<N, D, Key, Value>> {
        if depth > 0 {
            let children = &mut *self.node.inner;
            let insert_child = select::minimal_volume_increase(children, &entry.bounds()).unwrap();
            if let Some(new_node_ref) = insert_child.insert_entry(max_children, depth - 1, entry) {
                children.push(new_node_ref);
                if children.len() <= max_children {
                    None
                } else {
                    let (self_bounds, new_bounds, new_children) =
                        split::quadratic(max_children, children);
                    self.bounds = self_bounds;
                    Some(NodeRef {
                        bounds: new_bounds,
                        node: Node {
                            inner: ManuallyDrop::new(new_children),
                        },
                    })
                }
            } else {
                None
            }
        } else {
            match entry {
                NodeEntry::Inner(entry) => {
                    let children = &mut *self.node.inner;
                    // TODO: Avoid pushing if children are at capacity?
                    children.push(entry);
                    if children.len() <= max_children {
                        None
                    } else {
                        let (self_bounds, new_bounds, new_children) =
                            split::quadratic(max_children, children);
                        self.bounds = self_bounds;
                        Some(NodeRef {
                            bounds: new_bounds,
                            node: Node {
                                inner: ManuallyDrop::new(new_children),
                            },
                        })
                    }
                }
                NodeEntry::Leaf(entry) => {
                    let children = &mut *self.node.leaf;
                    // TODO: Avoid pushing if children are at capacity?
                    children.push(entry);
                    if children.len() <= max_children {
                        None
                    } else {
                        let (self_bounds, new_bounds, new_children) =
                            split::quadratic(max_children, children);
                        self.bounds = self_bounds;
                        Some(NodeRef {
                            bounds: new_bounds,
                            node: Node {
                                leaf: ManuallyDrop::new(new_children),
                            },
                        })
                    }
                }
            }
        }
    }

    unsafe fn recompute_bounds(&mut self, level: usize) {
        self.bounds = self.node.compute_bounds(level).unwrap();
    }
}

pub struct Config {
    pub max_children: usize,
    pub min_children: usize,
}

pub struct RTree<N, const D: usize, Key, Value> {
    config: Config,
    height: usize,
    root: MaybeUninit<NodeRef<N, D, Key, Value>>,
}

impl<N, const D: usize, Key, Value> Drop for RTree<N, D, Key, Value> {
    fn drop(&mut self) {
        if self.height > 0 {
            unsafe {
                self.root.assume_init_read().node.drop(self.height - 1);
            }
        }
    }
}

impl<'a, N, const D: usize, Key, Value> IntoIterator for &'a RTree<N, D, Key, Value>
where
    Key: Bounded<N, D>,
{
    type Item = &'a (Key, Value);

    type IntoIter = iter::Iter<'a, N, D, Key, Value>;

    fn into_iter(self) -> Self::IntoIter {
        if self.height > 0 {
            let root = unsafe { self.root.assume_init_ref() };
            unsafe { Self::IntoIter::new(self.height, &root.node) }
        } else {
            Self::IntoIter::empty()
        }
    }
}

impl<N, const D: usize, Key, Value> RTree<N, D, Key, Value>
where
    N: Ord,
    Key: Bounded<N, D>,
{
    pub fn iter<'a>(&'a self) -> iter::Iter<'a, N, D, Key, Value> {
        self.into_iter()
    }

    pub fn filter_iter<'a, Filter: SpatialFilter<N, D, Key>>(
        &'a self,
        filter: Filter,
    ) -> FilterIter<'a, N, D, Key, Value, Filter> {
        if self.height > 0 {
            let root = unsafe { self.root.assume_init_ref() };
            if filter.test_bounds(&root.bounds) {
                unsafe { FilterIter::new(self.height, &root.node, filter) }
            } else {
                FilterIter::empty(filter)
            }
        } else {
            FilterIter::empty(filter)
        }
    }

    pub fn new(config: Config) -> RTree<N, D, Key, Value> {
        return RTree {
            config,
            height: 0,
            root: MaybeUninit::uninit(),
        };
    }
}

impl<N, const D: usize, Key, Value> RTree<N, D, Key, Value>
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
    Key: Bounded<N, D>,
{
    /// Inserts an entry into an inner node at the given level.
    unsafe fn insert_entry(&mut self, level: usize, entry: NodeEntry<N, D, Key, Value>) {
        if self.height == 0 {
            panic!("Cannot insert entry into empty tree");
        }
        let root = unsafe { self.root.assume_init_mut() };
        if let Some(new_node_ref) =
            unsafe { root.insert_entry(self.config.max_children, self.height - level, entry) }
        {
            let prev_root = unsafe { self.root.assume_init_read() };
            self.root.write(NodeRef {
                bounds: min_bounds(&prev_root.bounds, &new_node_ref.bounds),
                node: Node {
                    inner: ManuallyDrop::new(vec![prev_root, new_node_ref]),
                },
            });
            self.height += 1;
        }
    }

    pub fn insert(&mut self, key: Key, value: Value) {
        if self.height > 0 {
            let root = unsafe { self.root.assume_init_mut() };
            if let Some(new_node_ref) = unsafe {
                root.insert_entry(
                    self.config.max_children,
                    self.height - 1,
                    NodeEntry::Leaf((key, value)),
                )
            } {
                let prev_root = unsafe { self.root.assume_init_read() };
                self.root.write(NodeRef {
                    bounds: min_bounds(&prev_root.bounds, &new_node_ref.bounds),
                    node: Node {
                        inner: ManuallyDrop::new(vec![prev_root, new_node_ref]),
                    },
                });
                self.height += 1;
            }
        } else {
            let bounds = key.bounds();
            let mut children = Vec::with_capacity(self.config.max_children);
            children.push((key, value));
            self.root.write(NodeRef {
                bounds,
                node: Node {
                    leaf: ManuallyDrop::new(children),
                },
            });
            self.height = 1;
        }
    }

    pub fn remove(&mut self, key: &Key, value: &Value) -> bool
    where
        Key: Eq,
        Value: Eq,
    {
        if self.height > 0 {
            let mut reinsert_nodes: Box<[Option<Node<N, D, Key, Value>>]> =
                std::iter::repeat_with(|| None)
                    .take(self.height - 1)
                    .collect();
            if unsafe {
                self.root.assume_init_mut().node.remove(
                    self.config.min_children,
                    self.height - 1,
                    key,
                    value,
                    &mut reinsert_nodes,
                )
            } {
                let root_len = unsafe { self.root.assume_init_ref().node.len(self.height - 1) };
                if root_len == 0 {
                    // Root has become empty and should be dropped
                    unsafe {
                        self.root.assume_init_read().node.drop(self.height - 1);
                    }
                    self.height = 0;
                } else if root_len == 1 {
                    if self.height > 1 {
                        // Root is an inner node with only one child
                        // The one child becomes the new root
                        let new_root =
                            (*unsafe { &mut self.root.assume_init_mut().node.inner }).remove(0);

                        // ensure old root is dropped before being overwritten
                        unsafe {
                            ManuallyDrop::drop(&mut self.root.assume_init_mut().node.inner);
                        }
                        *unsafe { self.root.assume_init_mut() } = new_root;
                        self.height -= 1;
                    } else {
                        // Root is a leaf node with only one child
                        // Recompute root bounds
                        unsafe {
                            self.root
                                .assume_init_mut()
                                .recompute_bounds(self.height - 1);
                        }
                    }
                } else {
                    // Root is an inner node with more than one child
                    // Recompute root bounds
                    unsafe {
                        self.root
                            .assume_init_mut()
                            .recompute_bounds(self.height - 1);
                    }
                }

                // reinsert nodes at leaf level
                if self.height > 0 {
                    if let Some(node) = reinsert_nodes[0].take() {
                        let children = ManuallyDrop::into_inner(unsafe { node.leaf });
                        for child in children {
                            unsafe {
                                self.insert_entry(0, NodeEntry::Leaf(child));
                            }
                        }
                    }
                }

                // reinsert nodes at inner levels
                for level in 1..self.height - 1 {
                    if let Some(node) = reinsert_nodes[level].take() {
                        let children = ManuallyDrop::into_inner(unsafe { node.inner });
                        for child in children {
                            unsafe {
                                self.insert_entry(level, NodeEntry::Inner(child));
                            }
                        }
                    }
                }

                return true;
            }
            return false;
        } else {
            false
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
    use crate::Config;

    use super::bounds::Bounds;
    use super::RTree;

    #[test]
    fn insert() {
        let mut tree = RTree::<i32, 2, Bounds<i32, 2>, ()>::new(Config {
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

    fn do_insert_bench(bencher: &mut Bencher, max_children: usize) {
        let min_children = max_children / 2;
        let mut rng = StdRng::from_seed([
            0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD,
            0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF,
            0xDE, 0xAD, 0xBE, 0xEF,
        ]);

        let mut tree = RTree::<i32, 2, Bounds<i32, 2>, i32>::new(Config {
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
        let mut tree = RTree::<i32, 2, Bounds<i32, 2>, i32>::new(Config {
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
        let mut stars = RTree::<N32, 3, Vector<N32, 3>, StarInfo>::new(Config {
            min_children: 4,
            max_children: 2,
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

        let space: Sphere<N32, 3> = Sphere {
            center: sol_pos,
            radius: n32(100.0),
        };
        let bounds: Bounds<N32, 3> = Bounds {
            min: sol_pos.into_map(|coord| coord - 100.0),
            max: sol_pos.into_map(|coord| coord + 100.0),
        };

        let mut star_lines = RTree::<N32, 3, Line<N32, 3>, ()>::new(Config {
            min_children: 4,
            max_children: 2,
        });

        for (pos, info) in stars.filter_iter(BoundedIntersectionFilter::new(space)) {
            if info.proper.len() > 0 {
                println!("{}", info.proper);
            }

            star_lines.insert(Line(sol_pos, *pos), ())
        }

        Ok(())
    }
}
