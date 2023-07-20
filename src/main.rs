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

use bounds::{empty_bounds, min_bounds, min_bounds_all, Bounded, Bounds};
use filter::SpatialFilter;
use intersects::Intersects;
use iter::FilterIter;
use std::{
    fmt::Debug,
    mem::{replace, take, ManuallyDrop},
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

struct LNodeRef<'a, N, const D: usize, Key, Value> {
    level: usize,
    value: &'a NodeRef<N, D, Key, Value>,
}

impl<'a, N, const D: usize, Key, Value> LNodeRef<'a, N, D, Key, Value> {
    fn debug_assert_bvh(&self) -> Bounds<N, D>
    where
        Key: Bounded<N, D>,
        N: Ord + num_traits::Bounded + Clone + Eq + Debug,
    {
        let bounds = if self.level > 0 {
            min_bounds_all(unsafe { &self.value.node.inner }.iter().map(|child| {
                LNodeRef {
                    level: self.level - 1,
                    value: child,
                }
                .debug_assert_bvh()
            }))
        } else {
            min_bounds_all(
                unsafe { &self.value.node.leaf }
                    .iter()
                    .map(|(key, _)| key.bounds()),
            )
        };
        assert_eq!(bounds, self.value.bounds);
        bounds
    }
}

impl<'a, N, const D: usize, Key, Value> Debug for LNodeRef<'a, N, D, Key, Value>
where
    N: Debug,
    Key: Debug,
    Value: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe { self.value.fmt(f, self.level) }
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
                child.drop(level - 1);
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

    unsafe fn take(&mut self, level: usize) -> Node<N, D, Key, Value> {
        if level > 0 {
            Node {
                inner: ManuallyDrop::new(take(&mut *self.inner)),
            }
        } else {
            Node {
                leaf: ManuallyDrop::new(take(&mut *self.leaf)),
            }
        }
    }

    unsafe fn fmt(&self, f: &mut std::fmt::Formatter<'_>, level: usize) -> std::fmt::Result
    where
        N: Debug,
        Key: Debug,
        Value: Debug,
    {
        if level > 0 {
            f.debug_list()
                .entries(unsafe {
                    self.inner.iter().map(|child| LNodeRef {
                        level: level - 1,
                        value: child,
                    })
                })
                .finish()
        } else {
            f.debug_list()
                .entries(unsafe { self.leaf.iter().map(|(key, value)| (key, value)) })
                .finish()
        }
    }
}

struct LNode<'a, N, const D: usize, Key, Value> {
    level: usize,
    value: &'a Node<N, D, Key, Value>,
}

impl<'a, N, const D: usize, Key, Value> Debug for LNode<'a, N, D, Key, Value>
where
    N: Debug,
    Key: Debug,
    Value: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe { self.value.fmt(f, self.level) }
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

impl<N, const D: usize, Key, Value> NodeRef<N, D, Key, Value> {
    unsafe fn drop(&mut self, level: usize) {
        self.node.drop(level);
    }

    fn take(&mut self, level: usize) -> NodeRef<N, D, Key, Value>
    where
        N: num_traits::Bounded,
    {
        NodeRef {
            bounds: replace(&mut self.bounds, empty_bounds()),
            node: unsafe { self.node.take(level) },
        }
    }

    unsafe fn fmt(&self, f: &mut std::fmt::Formatter<'_>, level: usize) -> std::fmt::Result
    where
        N: Debug,
        Key: Debug,
        Value: Debug,
    {
        f.debug_struct("NodeRef")
            .field("bounds", &self.bounds)
            .field(
                "node",
                &LNode {
                    level,
                    value: &self.node,
                },
            )
            .finish()
    }
}

impl<N, const D: usize, Key, Value> NodeRef<N, D, Key, Value>
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
    Key: Bounded<N, D>,
{
    /// Safety:
    /// - Node must be an inner node
    unsafe fn take_single_inner_child(&mut self) -> Option<NodeRef<N, D, Key, Value>>
    where
        N: num_traits::Bounded,
    {
        if self.node.inner.len() == 1 {
            self.bounds = empty_bounds();
            Some((*self.node.inner).swap_remove(0))
        } else {
            None
        }
    }

    /// Safety:
    /// - `depth` must be no greater than the height of the tree.
    /// - `entry` must be `NodeEntry::Inner` if nodes at the target `depth` are inner nodes, or `NodeEntry::Leaf` if they are leaf nodes.
    unsafe fn insert_entry(
        &mut self,
        max_children: usize,
        min_children: usize,
        depth: usize,
        entry: NodeEntry<N, D, Key, Value>,
    ) -> Option<NodeRef<N, D, Key, Value>> {
        let entry_bounds = entry.bounds();
        if depth > 0 {
            let children = &mut *self.node.inner;
            let insert_child = select::minimal_volume_increase(children, &entry.bounds()).unwrap();
            if let Some(new_node_ref) =
                insert_child.insert_entry(max_children, min_children, depth - 1, entry)
            {
                // TODO: Avoid pushing if children are at capacity?
                children.push(new_node_ref);
                if children.len() <= max_children {
                    self.bounds = min_bounds(&self.bounds, &entry_bounds);
                    None
                } else {
                    let (self_bounds, new_bounds, new_children) =
                        split::quadratic(max_children, min_children, children);
                    self.bounds = self_bounds;
                    Some(NodeRef {
                        bounds: new_bounds,
                        node: Node {
                            inner: ManuallyDrop::new(new_children),
                        },
                    })
                }
            } else {
                self.bounds = min_bounds(&self.bounds, &entry_bounds);
                None
            }
        } else {
            match entry {
                NodeEntry::Inner(entry) => {
                    let children = &mut *self.node.inner;
                    // TODO: Avoid pushing if children are at capacity?
                    children.push(entry);
                    if children.len() <= max_children {
                        self.bounds = min_bounds(&self.bounds, &entry_bounds);
                        None
                    } else {
                        let (self_bounds, new_bounds, new_children) =
                            split::quadratic(max_children, min_children, children);
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
                        self.bounds = min_bounds(&self.bounds, &entry_bounds);
                        None
                    } else {
                        let (self_bounds, new_bounds, new_children) =
                            split::quadratic(max_children, min_children, children);
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

    unsafe fn remove(
        &mut self,
        min_children: usize,
        level: usize,
        key: &Key,
        value: &Value,
        reinsert_nodes: &mut [Option<Node<N, D, Key, Value>>],
    ) -> bool
    where
        N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
        Key: Bounded<N, D> + Eq,
        Value: Eq,
    {
        if level > 0 {
            let children = &mut *self.node.inner;
            let mut i = children.len();
            while i > 0 {
                i -= 1;
                if children[i].bounds.intersects(&key.bounds())
                    && children[i].remove(min_children, level - 1, key, value, reinsert_nodes)
                {
                    if children[i].node.len(level - 1) < min_children {
                        let removed_child = children.swap_remove(i);
                        reinsert_nodes[level - 1] = Some(removed_child.node);
                    }

                    self.bounds = min_bounds_all(children.iter().map(|child| child.bounds()));

                    return true;
                }
            }
            return false;
        } else {
            let children = &mut *self.node.leaf;
            let index = children.iter().position(|(k, v)| k == key && v == value);
            if let Some(i) = index {
                children.remove(i);
                self.bounds = min_bounds_all(children.iter().map(|(key, _)| key.bounds()));

                return true;
            }
            return false;
        }
    }
}

#[derive(Debug)]
pub struct Config {
    pub max_children: usize,
    pub min_children: usize,
}

pub struct RTree<N, const D: usize, Key, Value> {
    config: Config,
    height: usize,
    root: NodeRef<N, D, Key, Value>,
}

impl<N, const D: usize, Key, Value> Debug for RTree<N, D, Key, Value>
where
    N: Debug,
    Key: Debug,
    Value: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RTree")
            .field("config", &self.config)
            .field(
                "root",
                &LNodeRef {
                    level: self.height,
                    value: &self.root,
                },
            )
            .finish()
    }
}

impl<N, const D: usize, Key, Value> Drop for RTree<N, D, Key, Value> {
    fn drop(&mut self) {
        unsafe {
            self.root.drop(self.height);
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
        unsafe { Self::IntoIter::new(self.height, &self.root.node) }
    }
}

impl<N, const D: usize, Key, Value> RTree<N, D, Key, Value>
where
    N: Ord + num_traits::Bounded,
    Key: Bounded<N, D>,
{
    pub fn iter<'a>(&'a self) -> iter::Iter<'a, N, D, Key, Value> {
        self.into_iter()
    }

    pub fn filter_iter<'a, Filter: SpatialFilter<N, D, Key>>(
        &'a self,
        filter: Filter,
    ) -> FilterIter<'a, N, D, Key, Value, Filter> {
        if filter.test_bounds(&self.root.bounds) {
            return unsafe { FilterIter::new(self.height, &self.root.node, filter) };
        }
        return FilterIter::empty(filter);
    }

    pub fn new(config: Config) -> RTree<N, D, Key, Value> {
        return RTree {
            height: 0,
            root: NodeRef {
                bounds: empty_bounds(),
                node: Node {
                    leaf: ManuallyDrop::new(Vec::with_capacity(config.max_children)),
                },
            },
            config,
        };
    }
}

impl<N, const D: usize, Key, Value> RTree<N, D, Key, Value>
where
    N: Ord + num_traits::Bounded + Clone + Sub<Output = N> + Into<f64>,
    Key: Bounded<N, D>,
{
    fn set_root(&mut self, new_root: NodeRef<N, D, Key, Value>) {
        unsafe {
            self.root.drop(self.height);
        }
        self.root = new_root;
    }

    /// Inserts an entry into a node at the given level.
    unsafe fn insert_entry(&mut self, level: usize, entry: NodeEntry<N, D, Key, Value>) {
        if let Some(new_node_ref) = unsafe {
            self.root.insert_entry(
                self.config.max_children,
                self.config.min_children,
                self.height - level,
                entry,
            )
        } {
            let bounds = min_bounds(&self.root.bounds, &new_node_ref.bounds);
            let mut next_root_children: Vec<NodeRef<N, D, Key, Value>> =
                Vec::with_capacity(self.config.max_children);
            next_root_children.push(self.root.take(self.height));
            next_root_children.push(new_node_ref);
            self.set_root(NodeRef {
                bounds,
                node: Node {
                    inner: ManuallyDrop::new(next_root_children),
                },
            });
            self.height += 1;
        }
    }

    pub fn insert(&mut self, key: Key, value: Value) {
        unsafe {
            self.insert_entry(0, NodeEntry::Leaf((key, value)));
        }
    }

    pub fn remove(&mut self, key: &Key, value: &Value) -> bool
    where
        Key: Eq,
        Value: Eq,
    {
        let height = self.height;
        let mut reinsert_nodes: Box<[Option<Node<N, D, Key, Value>>]> =
            std::iter::repeat_with(|| None).take(height).collect();
        if unsafe {
            self.root.remove(
                self.config.min_children,
                height,
                key,
                value,
                &mut reinsert_nodes,
            )
        } {
            // If root is an inner node with only one child, that child becomes the new root
            if height > 0 {
                if let Some(new_root) = unsafe { self.root.take_single_inner_child() } {
                    self.set_root(new_root);
                    self.height -= 1;
                }
            }

            // reinsert entries at leaf level
            if height > 0 {
                if let Some(node) = reinsert_nodes[0].take() {
                    let children = ManuallyDrop::into_inner(unsafe { node.leaf });
                    for leaf_entry in children.into_iter() {
                        unsafe {
                            self.insert_entry(0, NodeEntry::Leaf(leaf_entry));
                        }
                    }
                }
            }

            // reinsert entries at inner levels
            for level in 1..height {
                if let Some(node) = reinsert_nodes[level].take() {
                    let children = ManuallyDrop::into_inner(unsafe { node.inner });
                    for inner_entry in children.into_iter() {
                        unsafe {
                            self.insert_entry(level, NodeEntry::Inner(inner_entry));
                        }
                    }
                }
            }

            return true;
        }
        return false;
    }

    fn debug_assert_bvh(&self)
    where
        N: Debug,
    {
        LNodeRef {
            level: self.height,
            value: &self.root,
        }
        .debug_assert_bvh();
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

    #[test]
    fn remove() {
        let mut tree: RTree<i32, 2, Bounds<i32, 2>, i32> =
            RTree::<i32, 2, Bounds<i32, 2>, i32>::new(Config {
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

        {
            let mut rng = StdRng::from_seed([
                0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD,
                0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF,
                0xDE, 0xAD, 0xBE, 0xEF,
            ]);
            for i in 0..1000 {
                let min = Vector([rng.gen_range(0..991), rng.gen_range(0..991)]);
                let max = min + Vector([rng.gen_range(1..11), rng.gen_range(1..11)]);
                assert!(tree.remove(&Bounds { min, max }, &i));
                tree.debug_assert_bvh();
            }
        }

        unsafe {
            assert!(tree.root.node.inner.is_empty());
        }
    }

    fn do_insert_bench(bencher: &mut Bencher, max_children: usize) {
        let min_children = max_children / 4;
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
