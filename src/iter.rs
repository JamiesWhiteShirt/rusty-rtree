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

// TODO: Is Value constraint necessary?
pub trait SpatialFilter<N: Ord, const D: usize, Value: Bounded<N, D>> {
    fn test_bounds(&self, bounds: &Bounds<N, D>) -> bool;
    fn test_value(&self, value: &Value) -> bool;
}

pub struct SpatialFilterIter<'a, N: Ord, const D: usize, Value: Bounded<N, D>, Filter: SpatialFilter<N, D, Value>> {
    pub(crate) filter: Filter,
    pub(crate) tail: Vec<slice::Iter<'a, NodeRef<N, D, Value>>>,
    pub(crate) head: slice::Iter<'a, Value>,
}

impl<'a, N: Ord, const D: usize, Value: Bounded<N, D>, Filter: SpatialFilter<N, D, Value>> Iterator
    for SpatialFilterIter<'a, N, D, Value, Filter>
{
    type Item = &'a Value;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(value) = self
                .head
                .find(|value| self.filter.test_value(value))
            {
                return Some(value);
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
