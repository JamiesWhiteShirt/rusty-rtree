use std::slice;

use crate::{
    bounds::{Bounded},
    Node, NodeRef, filter::{SpatialFilter},
};

pub struct Iter<'a, N, const D: usize, Key, Value> {
    pub(crate) tail: Vec<slice::Iter<'a, NodeRef<N, D, Key, Value>>>,
    pub(crate) head: slice::Iter<'a, (Key, Value)>,
}

impl<'a, N, const D: usize, Key, Value> Iterator for Iter<'a, N, D, Key, Value> {
    type Item = &'a (Key, Value);

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

pub struct FilterIter<'a, N: Ord, const D: usize, Key, Value, Filter: SpatialFilter<N, D, Key>> {
    pub(crate) filter: Filter,
    pub(crate) tail: Vec<slice::Iter<'a, NodeRef<N, D, Key, Value>>>,
    pub(crate) head: slice::Iter<'a, (Key, Value)>,
}

impl<'a, N: Ord, const D: usize, Key: Bounded<N, D>, Value, Filter: SpatialFilter<N, D, Key>> Iterator
    for FilterIter<'a, N, D, Key, Value, Filter>
{
    type Item = &'a (Key, Value);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(item) = self
                .head
                .find(|(key, _)| self.filter.test_key(key))
            {
                return Some(item);
            }

            match self.tail.last_mut() {
                Some(tail_iter) => {
                    if let Some(intersecting) = tail_iter.find(|entry| self.filter.test_bounds(&entry.bounds)) {
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
