use std::{
    cmp,
    mem::replace,
    ops::{Mul, Sub},
};
mod bounds;
mod iter;
mod select;
mod split;

use bounds::{min_bounds, Bounded, Bounds};
use iter::IntersectionIter;

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

impl<N: Ord + Copy + Sub<Output = N> + Mul<Output = N>, const D: usize, Value: Bounded<N, D>>
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

impl<'a, N: Ord, const D: usize, Value: Bounded<N, D>> IntoIterator for &'a RTree<N, D, Value> {
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

impl<N: Ord + Copy + Sub<Output = N> + Mul<Output = N>, const D: usize, Value: Bounded<N, D>>
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
    use std::error::Error;
    use std::fs::File;
    use std::ops::Mul;
    use std::ops::Sub;

    use float_ord::FloatOrd;

    use super::bounds::Bounded;
    use super::bounds::Bounds;
    use super::RTree;

    struct F32Ord(pub FloatOrd<f32>);

    impl Mul for F32Ord {
        type Output = F32Ord;

        fn mul(self, rhs: Self) -> Self::Output {
            F32Ord(FloatOrd((self.0).0 * (rhs.0).0))
        }
    }

    impl Sub for F32Ord {
        type Output = F32Ord;

        fn sub(self, rhs: Self) -> Self::Output {
            F32Ord(FloatOrd((self.0).0 - (rhs.0).0))
        }
    }

    struct Star {
        id: u32,
        proper: Option<String>,
        x: f32,
        y: f32,
        z: f32,
    }

    impl Bounded<F32Ord, 3> for Star {
        fn bounds(&self) -> Bounds<F32Ord, 3> {
            return Bounds {
                min: [
                    F32Ord(FloatOrd(self.x)),
                    F32Ord(FloatOrd(self.y)),
                    F32Ord(FloatOrd(self.z)),
                ],
                max: [
                    F32Ord(FloatOrd(self.x)),
                    F32Ord(FloatOrd(self.y)),
                    F32Ord(FloatOrd(self.z)),
                ],
            };
        }
    }

    /* #[derive(Debug, serde::Deserialize)]
    struct Star {
        id: String,
        hip: Option<String>,
        hd: Option<String>,
        hr: Option<String>,
        gl: Option<String>,
        bf: Option<String>,
        proper: Option<String>,
        ra: f64,
        dec: f64,
        dist: f64,
        pmra: f64,
        pmdec: f64,
        rv: Option<f64>,
        mag: f64,
        absmag: f64,
        spect: Option<String>,
        ci: Option<f64>,
        x: f64,
        y: f64,
        z: f64,
        vx: f64,
        vy: f64,
        vz: f64,
        rarad: f64,
        decrad: f64,
        pmrarad: f64,
        pmdecrad: f64,
        // bayer,flam,con,comp,comp_primary,base,lum,var,var_min,var_max
    } */

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

    #[test]
    fn astronomy() -> Result<(), Box<dyn Error>> {
        let mut tree = RTree::<F32Ord, 3, Star>::new();

        let file = File::open("./hygdata_v3.csv");
        let mut rdr = csv::Reader::from_reader(file?);
        for result in rdr.records() {
            let record = result?;
            let id: u32 = record[0].parse()?;
            let proper: Option<String> = record[6].parse().ok();
            let x: f32 = record[17].parse()?;
            let y: f32 = record[18].parse()?;
            let z: f32 = record[19].parse()?;

            tree.insert(Star {
                id,
                proper,
                x,
                y,
                z,
            });
        }
        Ok(())
    }
}
