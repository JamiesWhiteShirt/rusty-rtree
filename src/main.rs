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
use std::{mem::replace, ops::Sub};

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

pub(crate) enum Node<N, const D: usize, Key, Value> {
    Inner(Vec<NodeRef<N, D, Key, Value>>),
    Leaf(Vec<(Key, Value)>),
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

enum RemoveResult {
    None,
    Removed,
    Underfull,
}

impl<N, const D: usize, Key, Value> NodeRef<N, D, Key, Value>
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
    Key: Bounded<N, D>,
{
    pub fn insert(
        &mut self,
        max_children: usize,
        key: Key,
        value: Value,
    ) -> Option<NodeRef<N, D, Key, Value>> {
        let bounds = key.bounds();
        self.bounds = min_bounds(&self.bounds, &bounds);
        match &mut self.node {
            Node::Inner(children) => {
                if children.len() == 0 {
                    panic!("Node has no children")
                }

                // TODO: Make selector configurable?
                let insert_child = select::minimal_volume_increase(children, &bounds).unwrap();
                if let Some(new_node_ref) = insert_child.insert(max_children, key, value) {
                    children.push(new_node_ref);
                    if children.len() <= max_children {
                        None
                    } else {
                        let (self_bounds, new_bounds, new_children) =
                            split::quadratic(max_children, children);
                        self.bounds = self_bounds;
                        Some(NodeRef {
                            bounds: new_bounds,
                            node: Node::Inner(new_children),
                        })
                    }
                } else {
                    None
                }
            }
            Node::Leaf(children) => {
                // TODO: Avoid pushing if children are at capacity?
                children.push((key, value));
                if children.len() <= max_children {
                    None
                } else {
                    let (self_bounds, new_bounds, new_children) =
                        split::quadratic(max_children, children);
                    self.bounds = self_bounds;
                    Some(NodeRef {
                        bounds: new_bounds,
                        node: Node::Leaf(new_children),
                    })
                }
            }
        }
    }

    fn remove(
        &mut self,
        min_children: usize,
        key: &Key,
        value: &Value,
        reinsert_inners: &mut Vec<NodeRef<N, D, Key, Value>>,
        reinsert_leafs: &mut Vec<(Key, Value)>,
    ) -> RemoveResult
    where
        Key: Eq,
        Value: Eq,
    {
        match &mut self.node {
            Node::Inner(children) => {
                let mut i = children.len();
                let mut removed = false;
                while i > 0 && !removed {
                    i -= 1;
                    if children[i].bounds.intersects(&key.bounds()) {
                        match children[i].remove(
                            min_children,
                            key,
                            value,
                            reinsert_inners,
                            reinsert_leafs,
                        ) {
                            RemoveResult::None => {}
                            RemoveResult::Removed => {
                                removed = true;
                            }
                            RemoveResult::Underfull => {
                                children.remove(i);
                                removed = true;
                            }
                        }
                    }
                }

                if children.len() < min_children {
                    reinsert_inners.append(children);
                    return RemoveResult::Underfull;
                }

                if removed {
                    self.bounds =
                        min_bounds_all(children.iter().map(|child| child.bounds())).unwrap();
                    return RemoveResult::Removed;
                }
                return RemoveResult::None;
            }
            Node::Leaf(children) => {
                let index = children.iter().position(|(k, v)| k == key && v == value);
                if let Some(i) = index {
                    children.remove(i);

                    if children.len() < min_children {
                        reinsert_leafs.append(children);
                        return RemoveResult::Underfull;
                    }

                    self.bounds = min_bounds_all(children.iter().map(|(k, _)| k.bounds())).unwrap();
                    return RemoveResult::Removed;
                }
                return RemoveResult::None;
            }
        }
    }
}

pub struct Config {
    pub max_children: usize,
    pub min_children: usize,
}

pub struct RTree<N, const D: usize, Key, Value> {
    config: Config,
    root: Option<NodeRef<N, D, Key, Value>>,
    empty_slice: [(Key, Value); 0],
}

impl<'a, N, const D: usize, Key, Value> IntoIterator for &'a RTree<N, D, Key, Value>
where
    Key: Bounded<N, D>,
{
    type Item = &'a (Key, Value);

    type IntoIter = iter::Iter<'a, N, D, Key, Value>;

    fn into_iter(self) -> Self::IntoIter {
        if let Some(root) = &self.root {
            match &root.node {
                Node::Inner(children) => iter::Iter {
                    tail: vec![children.iter()],
                    head: self.empty_slice.iter(),
                },
                Node::Leaf(children) => iter::Iter {
                    tail: Vec::new(),
                    head: children.iter(),
                },
            }
        } else {
            iter::Iter {
                tail: Vec::new(),
                head: self.empty_slice.iter(),
            }
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
        if let Some(root) = &self.root {
            match &root.node {
                Node::Inner(children) => FilterIter {
                    filter,
                    tail: vec![children.iter()],
                    head: self.empty_slice.iter(),
                },
                Node::Leaf(children) => FilterIter {
                    filter,
                    tail: Vec::new(),
                    head: children.iter(),
                },
            }
        } else {
            FilterIter {
                filter,
                tail: Vec::new(),
                head: self.empty_slice.iter(),
            }
        }
    }

    pub fn new(config: Config) -> RTree<N, D, Key, Value> {
        return RTree {
            config,
            root: None,
            empty_slice: [],
        };
    }
}

impl<N, const D: usize, Key, Value> RTree<N, D, Key, Value>
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
    Key: Bounded<N, D>,
{
    pub fn insert(&mut self, key: Key, value: Value) {
        if let Some(root) = &mut self.root {
            if let Some(new_node_ref) = root.insert(self.config.max_children, key, value) {
                // TODO: Is there a better way to do this?
                let prev_root = replace(
                    root,
                    NodeRef {
                        bounds: new_node_ref.bounds.clone(),
                        node: Node::Inner(vec![new_node_ref]),
                    },
                );
                root.bounds = min_bounds(&root.bounds, &prev_root.bounds);
                if let Node::Inner(children) = &mut root.node {
                    children.push(prev_root);
                } else {
                    panic!("New root must be an inner node");
                }
            }
        } else {
            let bounds = key.bounds();
            let mut children = Vec::with_capacity(self.config.max_children);
            children.push((key, value));
            self.root = Some(NodeRef {
                bounds,
                node: Node::Leaf(children),
            })
        }
    }

    pub fn remove(&mut self, key: &Key, value: &Value) -> bool
    where
        Key: Eq,
        Value: Eq,
    {
        // TODO: Implement reinsertion
        let mut reinsert_inners: Vec<NodeRef<N, D, Key, Value>> = Vec::new();
        let mut reinsert_leafs: Vec<(Key, Value)> = Vec::new();
        if let Some(root) = &mut self.root {
            root.remove(
                self.config.min_children,
                &key,
                &value,
                &mut reinsert_inners,
                &mut reinsert_leafs,
            );
            true
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
