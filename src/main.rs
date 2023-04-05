mod bounds;
mod iter;
mod position;
mod select;
mod split;

use bounds::{min_bounds, Bounded, Bounds};
use iter::IntersectionIter;
use std::{mem::replace, ops::Sub};

fn main() {
    println!("Hello, world!");
}

pub(crate) struct NodeRef<N, const D: usize, Value: Bounded<N, D>> {
    pub(crate) bounds: Bounds<N, D>,
    pub(crate) node: Node<N, D, Value>,
}

impl<N: Copy, const D: usize, Value: Bounded<N, D>> Bounded<N, D> for NodeRef<N, D, Value> {
    fn bounds(&self) -> Bounds<N, D> {
        self.bounds
    }
}

pub(crate) enum Node<N, const D: usize, Value: Bounded<N, D>> {
    Inner(Vec<NodeRef<N, D, Value>>),
    Leaf(Vec<Value>),
}

// TODO: Make this configurable
const MAX_CHILDREN: usize = 4;
const MIN_CHILDREN: usize = 2;

impl<N: Ord + Copy + Sub<Output = N> + Into<f64>, const D: usize, Value: Bounded<N, D>>
    NodeRef<N, D, Value>
{
    pub fn insert(&mut self, value: Value) -> Option<NodeRef<N, D, Value>> {
        let bounds = value.bounds();
        self.bounds = min_bounds(&self.bounds, &bounds);
        match &mut self.node {
            Node::Inner(children) => {
                if children.len() == 0 {
                    panic!("Node has no children")
                }

                // TODO: Make selector configurable?
                let insert_child = select::minimal_volume_increase(children, &bounds).unwrap();
                if let Some(new_node_ref) = insert_child.insert(value) {
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
                children.push(value);
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

pub struct RTree<N, const D: usize, Value: Bounded<N, D>> {
    root: Option<NodeRef<N, D, Value>>,
    empty_slice: [Value; 0],
}

impl<'a, N, const D: usize, Value: Bounded<N, D>> IntoIterator for &'a RTree<N, D, Value> {
    type Item = &'a Value;

    type IntoIter = iter::Iter<'a, N, D, Value>;

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

impl<N: Ord, const D: usize, Value: Bounded<N, D>> RTree<N, D, Value> {
    pub fn iter<'a>(&'a self) -> iter::Iter<'a, N, D, Value> {
        self.into_iter()
    }

    pub fn get_intersecting<'a>(
        &'a self,
        bounds: Bounds<N, D>,
    ) -> IntersectionIter<'a, N, D, Value> {
        if let Some(root) = &self.root {
            match &root.node {
                Node::Inner(children) => IntersectionIter {
                    bounds: bounds,
                    tail: vec![children.iter()],
                    head: self.empty_slice.iter(),
                },
                Node::Leaf(children) => IntersectionIter {
                    bounds: bounds,
                    tail: Vec::new(),
                    head: children.iter(),
                },
            }
        } else {
            IntersectionIter {
                bounds: bounds,
                tail: Vec::new(),
                head: self.empty_slice.iter(),
            }
        }
    }

    fn new() -> RTree<N, D, Value> {
        return RTree {
            root: None,
            empty_slice: [],
        };
    }
}

impl<N: Ord + Copy + Sub<Output = N> + Into<f64>, const D: usize, Value: Bounded<N, D>>
    RTree<N, D, Value>
{
    pub fn insert(&mut self, value: Value) {
        if let Some(root) = &mut self.root {
            if let Some(new_node_ref) = root.insert(value) {
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
                bounds: value.bounds(),
                node: Node::Leaf(vec![value]),
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

    use super::bounds::Bounded;
    use super::bounds::Bounds;
    use super::RTree;

    struct Star {
        id: u32,
        proper: String,
        x: N32,
        y: N32,
        z: N32,
    }

    impl Bounded<N32, 3> for Star {
        fn bounds(&self) -> Bounds<N32, 3> {
            return Bounds {
                min: [self.x, self.y, self.z],
                max: [self.x, self.y, self.z],
            };
        }
    }

    #[test]
    fn insert() {
        let mut tree = RTree::<i32, 2, Bounds<i32, 2>>::new();

        tree.insert(Bounds {
            min: [0, 0],
            max: [1, 1],
        });

        tree.insert(Bounds {
            min: [2, 0],
            max: [3, 1],
        });

        tree.insert(Bounds {
            min: [0, 2],
            max: [1, 3],
        });

        tree.insert(Bounds {
            min: [2, 2],
            max: [3, 3],
        });

        tree.insert(Bounds {
            min: [0, 2],
            max: [0, 2],
        });
    }

    fn record_to_star(record: csv::StringRecord) -> Result<Star, Box<dyn Error>> {
        let id: u32 = record[0].parse()?;
        let proper: String = record[6].parse()?;
        let x: N32 = n32(record[17].parse()?);
        let y: N32 = n32(record[18].parse()?);
        let z: N32 = n32(record[19].parse()?);

        Ok(Star {
            id,
            proper,
            x,
            y,
            z,
        })
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
        let mut tree = RTree::<N32, 3, Star>::new();

        let file = File::open("./hygdata_v3.csv");
        let mut rdr = csv::Reader::from_reader(file?);
        let mut iter = rdr.records();

        let sol = record_to_star(iter.next().ok_or(Box::new(SolNotFoundError))??)?;
        tree.insert(sol);

        for result in iter {
            tree.insert(record_to_star(result?)?);
        }

        let bounds: Bounds<N32, 3> = Bounds {
            min: [n32(-100.0), n32(-100.0), n32(-100.0)],
            max: [n32(100.0), n32(100.0), n32(100.0)],
        };

        for star in tree.get_intersecting(bounds) {
            if star.proper.len() > 0 {
                println!("{}", star.proper);
            }
        }

        Ok(())
    }
}
