mod bounds;
mod filter;
mod intersects;
mod iter;
mod position;
mod select;
mod sphere;
mod split;

use bounds::{min_bounds, Bounded, Bounds};
use filter::SpatialFilter;
use iter::FilterIter;
use std::{mem::replace, ops::Sub};

fn main() {
    println!("Hello, world!");
}

pub(crate) struct NodeRef<N, const D: usize, Key, Value> {
    pub(crate) bounds: Bounds<N, D>,
    pub(crate) node: Node<N, D, Key, Value>,
}

impl<N: Copy, const D: usize, Key: Bounded<N, D>, Value> Bounded<N, D> for NodeRef<N, D, Key, Value> {
    fn bounds(&self) -> Bounds<N, D> {
        self.bounds
    }
}

pub(crate) enum Node<N, const D: usize, Key, Value> {
    Inner(Vec<NodeRef<N, D, Key, Value>>),
    Leaf(Vec<(Key, Value)>),
}

// TODO: Make this configurable
const MAX_CHILDREN: usize = 4;
const MIN_CHILDREN: usize = 2;

impl<N: Ord, const D: usize, Key: Bounded<N, D>, Value> Bounded<N, D> for (Key, Value) {
    fn bounds(&self) -> Bounds<N, D> {
        self.0.bounds()
    }
}

impl<N: Ord + Copy + Sub<Output = N> + Into<f64>, const D: usize, Key: Bounded<N, D>, Value>
    NodeRef<N, D, Key, Value>
{
    pub fn insert(&mut self, key: Key, value: Value) -> Option<NodeRef<N, D, Key, Value>> {
        let bounds = key.bounds();
        self.bounds = min_bounds(&self.bounds, &bounds);
        match &mut self.node {
            Node::Inner(children) => {
                if children.len() == 0 {
                    panic!("Node has no children")
                }

                // TODO: Make selector configurable?
                let insert_child = select::minimal_volume_increase(children, &bounds).unwrap();
                if let Some(new_node_ref) = insert_child.insert(key, value) {
                    children.push(new_node_ref);
                    if children.len() <= MAX_CHILDREN {
                        None
                    } else {
                        let (self_bounds, new_bounds, new_children) = split::quadratic(children);
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
                children.push((key, value));
                if children.len() <= MAX_CHILDREN {
                    None
                } else {
                    let (self_bounds, new_bounds, new_children) = split::quadratic(children);
                    self.bounds = self_bounds;
                    Some(NodeRef {
                        bounds: new_bounds,
                        node: Node::Leaf(new_children),
                    })
                }
            }
        }
    }
}

pub struct RTree<N, const D: usize, Key: Bounded<N, D>, Value> {
    root: Option<NodeRef<N, D, Key, Value>>,
    empty_slice: [(Key, Value); 0],
}

impl<'a, N, const D: usize, Key: Bounded<N, D>, Value> IntoIterator for &'a RTree<N, D, Key, Value> {
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

impl<N: Ord, const D: usize, Key: Bounded<N, D>, Value> RTree<N, D, Key, Value> {
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

    pub fn new() -> RTree<N, D, Key, Value> {
        return RTree {
            root: None,
            empty_slice: [],
        };
    }
}

impl<N: Ord + Copy + Sub<Output = N> + Into<f64>, const D: usize, Key: Bounded<N, D>, Value>
    RTree<N, D, Key, Value>
{
    pub fn insert(&mut self, key: Key, value: Value) {
        if let Some(root) = &mut self.root {
            if let Some(new_node_ref) = root.insert(key, value) {
                // TODO: Is there a better way to do this?
                let prev_root = replace(
                    root,
                    NodeRef {
                        bounds: new_node_ref.bounds,
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
            self.root = Some(NodeRef {
                bounds: key.bounds(),
                node: Node::Leaf(vec![(key, value)]),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use core::fmt;
    use std::error::Error;
    use std::fs::File;

    use noisy_float::types::n32;
    use noisy_float::types::N32;

    use crate::filter::BoundedIntersectionFilter;
    use crate::sphere::Sphere;

    use super::bounds::Bounds;
    use super::RTree;

    #[test]
    fn insert() {
        let mut tree = RTree::<i32, 2, Bounds<i32, 2>, ()>::new();

        tree.insert(Bounds {
            min: [0, 0],
            max: [1, 1],
        }, ());

        tree.insert(Bounds {
            min: [2, 0],
            max: [3, 1],
        }, ());

        tree.insert(Bounds {
            min: [0, 2],
            max: [1, 3],
        }, ());

        tree.insert(Bounds {
            min: [2, 2],
            max: [3, 3],
        }, ());

        tree.insert(Bounds {
            min: [0, 2],
            max: [0, 2],
        }, ());
    }

    struct StarInfo {
        id: u32,
        proper: String,
    }


    fn record_to_star(record: csv::StringRecord) -> Result<([N32; 3], StarInfo), Box<dyn Error>> {
        let id: u32 = record[0].parse()?;
        let proper: String = record[6].parse()?;
        let x: N32 = n32(record[17].parse()?);
        let y: N32 = n32(record[18].parse()?);
        let z: N32 = n32(record[19].parse()?);

        Ok(([x, y, z], StarInfo {
            id,
            proper,
        }))
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
        let mut tree = RTree::<N32, 3, [N32; 3], StarInfo>::new();

        let file = File::open("./hygdata_v3.csv");
        let mut rdr = csv::Reader::from_reader(file?);
        let mut iter = rdr.records();

        let (sol_pos, sol_info) = record_to_star(iter.next().ok_or(Box::new(SolNotFoundError))??)?;
        tree.insert(sol_pos, sol_info);

        for result in iter {
            let (pos, info) = record_to_star(result?)?;
            tree.insert(pos, info);
        }

        let space: Sphere<N32, 3> = Sphere { center: sol_pos, radius: n32(100.0) };
        let bounds: Bounds<N32, 3> = Bounds {
            min: sol_pos.map(|coord| coord - 100.0),
            max: sol_pos.map(|coord| coord + 100.0),
        };

        let filter = BoundedIntersectionFilter::new(space);

        for (_, info) in tree.filter_iter(filter) {
            if info.proper.len() > 0 {
                println!("{}", info.proper);
            }
        }

        Ok(())
    }
}
