use std::{cmp, slice};
mod bounds;
mod rtree;

use array_init::from_iter;

fn main() {
    println!("Hello, world!");
}

trait Bounded<N: Ord, const D: usize> {
    fn bounds(&self) -> Bounds<N, D>;
}

struct Bounds<N: Ord, const D: usize> {
    min: [N; D],
    max: [N; D],
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
}

struct NodeRef<N: Ord, const D: usize, Value: Bounded<N, D>> {
    bounds: Bounds<N, D>,
    node: Node<N, D, Value>,
}

enum Node<N: Ord, const D: usize, Value: Bounded<N, D>> {
    Inner(Vec<NodeRef<N, D, Value>>),
    Leaf(Vec<Value>),
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
        return Bounds { min, max };
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
}

#[cfg(test)]
mod tests {
    #[test]
    fn exploration() {
        assert_eq!(2 + 1, 3);
    }
}
