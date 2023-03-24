use std::{
    cmp::{self, Ordering},
    iter::repeat,
    mem::replace,
    ops::{Mul, Sub},
    slice,
};

use array_init::from_iter;

trait Bounded<N: Ord, const D: usize> {
    fn bounds(&self) -> Bounds<N, D>;
}

#[derive(Clone, Copy)]
struct Bounds<N: Ord, const D: usize> {
    min: [N; D],
    max: [N; D],
}

fn min_bounds<N: Ord + Copy, const D: usize>(
    lhs: &Bounds<N, D>,
    rhs: &Bounds<N, D>,
) -> Bounds<N, D> {
    let min = from_iter(
        lhs.min
            .into_iter()
            .zip(rhs.min)
            .map(|(lhs, rhs)| cmp::min(lhs, rhs)),
    )
    .unwrap();
    let max = from_iter(
        lhs.max
            .into_iter()
            .zip(rhs.max)
            .map(|(lhs, rhs)| cmp::max(lhs, rhs)),
    )
    .unwrap();
    Bounds { min, max }
}

impl<N: Ord + Copy, const D: usize> Bounded<N, D> for Bounds<N, D> {
    fn bounds(&self) -> Bounds<N, D> {
        *self
    }
}

impl<N: Ord, const D: usize> Bounds<N, D> {
    fn intersects(&self, rhs: &Self) -> bool {
        self.min
            .iter()
            .zip(rhs.max.iter())
            .all(|(lhs_min, rhs_max)| lhs_min <= rhs_max)
            && self
                .max
                .iter()
                .zip(rhs.min.iter())
                .all(|(lhs_max, rhs_min)| lhs_max >= rhs_min)
    }

    fn contains(&self, rhs: &Self) -> bool {
        self.min
            .iter()
            .zip(rhs.min.iter())
            .all(|(lhs_min, rhs_min)| lhs_min <= rhs_min)
            && self
                .max
                .iter()
                .zip(rhs.max.iter())
                .all(|(lhs_max, rhs_max)| lhs_max >= rhs_max)
    }
}

impl<N: Ord + Copy + Sub<Output = N> + Mul<Output = N>, const D: usize> Bounds<N, D> {
    fn volume(&self) -> N {
        // TODO: Is there a better way to handle this constraint?
        if D == 0 {
            panic!("Cannot calculate volume of bounds with D = 0")
        }

        self.min
            .iter()
            .zip(self.max.iter())
            .map(|(min, max)| *max - *min)
            .reduce(|acc, length| acc * length)
            .unwrap()
    }

    fn volume_increase_of_min_bounds(&self, other: &Self) -> N {
        min_bounds(self, other).volume() - self.volume()
    }
}

struct NodeRef<N: Ord, const D: usize, Value: Bounded<N, D>> {
    bounds: Bounds<N, D>,
    node: Node<N, D, Value>,
}

impl<N: Ord + Copy, const D: usize, Value: Bounded<N, D>> Bounded<N, D> for NodeRef<N, D, Value> {
    fn bounds(&self) -> Bounds<N, D> {
        self.bounds
    }
}

enum Node<N: Ord, const D: usize, Value: Bounded<N, D>> {
    Inner(Vec<NodeRef<N, D, Value>>),
    Leaf(Vec<Value>),
}

fn minimal_volume_increase_select<
    'a,
    N: Ord + Copy + Sub<Output = N> + Mul<Output = N>,
    const D: usize,
    Value: Bounded<N, D>,
>(
    children: &'a mut [Value],
    bounds: &Bounds<N, D>,
) -> Option<&'a mut Value> {
    children.into_iter().min_by(|lhs, rhs| {
        // Optimize for minimal volume increase
        let cmp = N::cmp(
            &lhs.bounds().volume_increase_of_min_bounds(bounds),
            &rhs.bounds().volume_increase_of_min_bounds(bounds),
        );
        if cmp == Ordering::Equal {
            // If the volume increase is the same, select the child with the smallest volume to start with
            N::cmp(&lhs.bounds().volume(), &rhs.bounds().volume())
        } else {
            cmp
        }
    })
}

/// Returns a pair of indices (a, b) where a < b. b is therefore also never zero.
fn worst_combination<
    N: Ord + Copy + Sub<Output = N> + Mul<Output = N>,
    const D: usize,
    Value: Bounded<N, D>,
>(
    children: &[Value],
) -> (usize, usize) {
    if children.len() < 2 {
        panic!("Must have more than 2 children!");
    }
    (0..children.len() - 1)
        .flat_map(|lhs| repeat(lhs).zip((lhs + 1)..children.len()))
        .max_by_key(|(lhs, rhs)| {
            min_bounds(&children[*lhs].bounds(), &children[*rhs].bounds()).volume()
        })
        .unwrap()
}

/// Seeds splitting of values into two groups by finding two values which will
/// form the seeds of two groups. The seed of the first group is moved to
/// values[0], while the seed of the second group is returned.
fn seed_split_groups<
    N: Ord + Copy + Sub<Output = N> + Mul<Output = N>,
    const D: usize,
    Value: Bounded<N, D>,
>(
    values: &mut Vec<Value>,
) -> Value {
    let (i_1, i_2) = worst_combination(values);
    values.swap(0, i_1);
    values.remove(i_2)
}

fn best_candidate_for_group<
    N: Ord + Copy + Sub<Output = N> + Mul<Output = N>,
    const D: usize,
    Value: Bounded<N, D>,
>(
    children: &[Value],
    bounds: &Bounds<N, D>,
) -> Option<(usize, N)> {
    children
        .into_iter()
        .enumerate()
        .map(|(i, value)| (i, min_bounds(&value.bounds(), bounds).volume()))
        .min_by_key(|(_, volume)| *volume)
}

/// Splits values into two groups. When it returns, values contains the values of
/// the first group while the other group is returned along with its minimum
/// bounds.
fn quadratic_split<
    N: Ord + Copy + Sub<Output = N> + Mul<Output = N>,
    const D: usize,
    Value: Bounded<N, D>,
>(
    values: &mut Vec<Value>,
) -> (Bounds<N, D>, Bounds<N, D>, Vec<Value>) {
    if values.len() < 2 {
        panic!("Must have more than 2 children to split!");
    }

    let min_group_size = values.len() / 2;

    let mut group_2 = vec![seed_split_groups(values)];
    let (mut bounds_1, mut bounds_2) = (values[0].bounds(), group_2[0].bounds());

    let mut group_1_len = 1;
    // children is now partitioned such that children[0..group_1_len] is group_1
    // and children[group_1_len..] is the remaining children to be distributed
    // into groups.
    // When the loop terminates, children is group_1.
    while group_1_len < values.len() {
        let remaining = &values[group_1_len..];
        let (candidate_1, candidate_2) = (
            best_candidate_for_group(remaining, &bounds_1).unwrap(),
            best_candidate_for_group(remaining, &bounds_2).unwrap(),
        );

        let add_to_group_1 = if candidate_1.1 < candidate_2.1 {
            group_2.len() + remaining.len() - 1 >= min_group_size
        } else {
            group_1_len + remaining.len() - 1 == min_group_size
        };

        if add_to_group_1 {
            bounds_1 = min_bounds(&bounds_1, &remaining[candidate_1.0].bounds());
            values.swap(group_1_len + candidate_1.0, group_1_len);
            group_1_len += 1;
        } else {
            bounds_2 = min_bounds(&bounds_2, &remaining[candidate_2.0].bounds());
            group_2.push(values.remove(candidate_2.0))
        }
    }

    (bounds_1, bounds_2, group_2)
}

// TODO: Make this configurable
const MAX_CHILDREN: usize = 4;
const MIN_CHILDREN: usize = 2;

impl<N: Ord + Copy + Sub<Output = N> + Mul<Output = N>, const D: usize, Value: Bounded<N, D>>
    NodeRef<N, D, Value>
{
    fn insert(&mut self, value: Value) -> Option<NodeRef<N, D, Value>> {
        let bounds = value.bounds();
        self.bounds = min_bounds(&self.bounds, &bounds);
        match &mut self.node {
            Node::Inner(children) => {
                if children.len() == 0 {
                    panic!("Node has no children")
                }

                // TODO: Make selector configurable?
                let insert_child = minimal_volume_increase_select(children, &bounds).unwrap();
                if let Some(new_node_ref) = insert_child.insert(value) {
                    children.push(new_node_ref);
                    if children.len() <= MAX_CHILDREN {
                        None
                    } else {
                        let (self_bounds, new_bounds, new_children) = quadratic_split(children);
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
                    let (self_bounds, new_bounds, new_children) = quadratic_split(children);
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

struct IntersectionIter<'a, N: Ord, const D: usize, Value: Bounded<N, D>> {
    bounds: Bounds<N, D>,
    tail: Vec<slice::Iter<'a, NodeRef<N, D, Value>>>,
    head: slice::Iter<'a, Value>,
}

impl<'a, N: Ord, const D: usize, Value: Bounded<N, D>> Iterator
    for IntersectionIter<'a, N, D, Value>
{
    type Item = &'a Value;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(value) = self
                .head
                .find(|value| value.bounds().intersects(&self.bounds))
            {
                return Some(value);
            }

            match self.tail.last_mut() {
                Some(node) => {
                    if let Some(entry) = node.find(|entry| entry.bounds.intersects(&self.bounds)) {
                        match &entry.node {
                            Node::Inner(entries) => self.tail.push(entries.iter()),
                            Node::Leaf(entries) => self.head = entries.iter(),
                        }
                    }
                }
                None => return Option::None,
            }
        }
    }
}

impl<N: Ord, const D: usize, Value: Bounded<N, D>> Node<N, D, Value> {
    /* fn get_values(&self, bounds: Bounds<N, Dim>) -> Vec<Value> {
        match self {
            Node::Inner(children) =>
                children
                    .into_iter()
                    .filter(|entry| entry.bounds.intersects(bounds))
                    .flat_map(|entry| entry.child.get_values(bounds))
                    .collect::<Vec<Value>>(),
            Node::Leaf(children) => children
                    .into_iter()
                    .filter(|value| bounds.intersects(value.bounds()))
                    .collect::<Vec<Value>>(),
        }
    } */
}

struct Line<N, const D: usize> {
    start: [N; D],
    end: [N; D],
}

impl<N: Ord + Copy, const D: usize> Bounded<N, D> for Line<N, D> {
    fn bounds(&self) -> Bounds<N, D> {
        let min = from_iter(
            self.start
                .into_iter()
                .zip(self.end)
                .map(|(start, end)| cmp::min(start, end)),
        )
        .unwrap();
        let max = from_iter(
            self.start
                .into_iter()
                .zip(self.end)
                .map(|(start, end)| cmp::max(start, end)),
        )
        .unwrap();
        Bounds { min, max }
    }
}

struct RTree<N: Ord, const D: usize, Value: Bounded<N, D>> {
    root: Option<NodeRef<N, D, Value>>,
    empty_slice: [Value; 0],
}

impl<N: Ord, const D: usize, Value: Bounded<N, D>> RTree<N, D, Value> {
    fn get_intersecting<'a>(&'a self, bounds: Bounds<N, D>) -> IntersectionIter<'a, N, D, Value> {
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
    fn insert(&mut self, value: Value) {
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
    use super::{Bounds, RTree};

    #[test]
    fn exploration() {
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
}
