use std::slice;

use crate::{
    bounds::{Bounded, Bounds},
    Node, NodeRef,
};

pub struct Iter<'a, N, const D: usize, Value: Bounded<N, D>> {
    pub(crate) tail: Vec<slice::Iter<'a, NodeRef<N, D, Value>>>,
    pub(crate) head: slice::Iter<'a, Value>,
}

impl<'a, N, const D: usize, Value: Bounded<N, D>> Iterator for Iter<'a, N, D, Value> {
    type Item = &'a Value;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(value) = self.head.next() {
                return Some(value);
            }

            match self.tail.last_mut() {
                Some(node) => {
                    if let Some(entry) = node.next() {
                        match &entry.node {
                            Node::Inner(entries) => self.tail.push(entries.iter()),
                            Node::Leaf(entries) => self.head = entries.iter(),
                        }
                    }
                }
                None => return Option::None,
            }

            // Clean up by removing empty iterators from the tail
            while self.tail.last().map(|tail_iter| tail_iter.len()) == Some(0) {
                self.tail.pop();
            }
        }
    }
}

pub struct IntersectionIter<'a, N: Ord, const D: usize, Value: Bounded<N, D>> {
    pub(crate) bounds: Bounds<N, D>,
    pub(crate) tail: Vec<slice::Iter<'a, NodeRef<N, D, Value>>>,
    pub(crate) head: slice::Iter<'a, Value>,
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
                Some(tail_iter) => {
                    if let Some(intersecting) = tail_iter.find(|entry| entry.bounds.intersects(&self.bounds)) {
                        match &intersecting.node {
                            Node::Inner(entries) => self.tail.push(entries.iter()),
                            Node::Leaf(entries) => self.head = entries.iter(),
                        }
                    }
                }
                None => return Option::None,
            }

            // Clean up by removing empty iterators from the tail
            while self.tail.last().map(|tail_iter| tail_iter.len()) == Some(0) {
                self.tail.pop();
            }
        }
    }
}
