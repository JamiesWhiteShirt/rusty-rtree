use std::{mem::ManuallyDrop, slice};

use crate::{bounds::Bounded, filter::SpatialFilter, Node, NodeRef};

union IterLevel<'a, N, const D: usize, Key, Value> {
    inner: ManuallyDrop<slice::Iter<'a, NodeRef<N, D, Key, Value>>>,
    leaf: ManuallyDrop<slice::Iter<'a, (Key, Value)>>,
}

pub struct Iter<'a, N, const D: usize, Key, Value> {
    // TODO: Use vector capacity instead of height?
    height: usize,
    stack: Vec<IterLevel<'a, N, D, Key, Value>>,
}

impl<'a, N, const D: usize, Key, Value> Iter<'a, N, D, Key, Value> {
    pub(crate) unsafe fn new(
        height: usize,
        root: &'a Node<N, D, Key, Value>,
    ) -> Iter<'a, N, D, Key, Value> {
        let mut stack = Vec::with_capacity(height);
        stack.push(if height > 1 {
            IterLevel {
                inner: ManuallyDrop::new(root.inner.iter()),
            }
        } else {
            IterLevel {
                leaf: ManuallyDrop::new(root.leaf.iter()),
            }
        });
        Iter { height, stack }
    }

    pub(crate) fn empty() -> Iter<'a, N, D, Key, Value> {
        Iter {
            height: 0,
            stack: Vec::new(),
        }
    }
}

impl<'a, N, const D: usize, Key, Value> Iterator for Iter<'a, N, D, Key, Value> {
    type Item = &'a (Key, Value);

    fn next(&mut self) -> Option<Self::Item> {
        while !self.stack.is_empty() {
            let level = self.height - self.stack.len();
            if level == 0 {
                // Iterating over leaf node
                if let Some(value) = (*unsafe { &mut self.stack.last_mut().unwrap().leaf }).next() {
                    return Some(value);
                } else {
                    unsafe { ManuallyDrop::drop(&mut self.stack.pop().unwrap().leaf) };
                }
            } else {
                // Iterating over inner node
                if let Some(entry) = (*unsafe { &mut self.stack.last_mut().unwrap().inner }).next()
                {
                    self.stack.push(if level > 1 {
                        IterLevel {
                            inner: ManuallyDrop::new((*unsafe { &entry.node.inner }).iter()),
                        }
                    } else {
                        IterLevel {
                            leaf: ManuallyDrop::new((*unsafe { &entry.node.leaf }).iter()),
                        }
                    })
                } else {
                    unsafe { ManuallyDrop::drop(&mut self.stack.pop().unwrap().inner) };
                }
            }
        }
        Option::None
    }
}

pub struct FilterIter<'a, N, const D: usize, Key, Value, Filter> {
    filter: Filter,
    // TODO: Use vector capacity instead of height?
    height: usize,
    stack: Vec<IterLevel<'a, N, D, Key, Value>>,
}

impl<'a, N, const D: usize, Key, Value, Filter> FilterIter<'a, N, D, Key, Value, Filter> {
    pub(crate) unsafe fn new(
        height: usize,
        root: &'a Node<N, D, Key, Value>,
        filter: Filter,
    ) -> FilterIter<'a, N, D, Key, Value, Filter> {
        let mut stack = Vec::with_capacity(height);
        stack.push(if height > 1 {
            IterLevel {
                inner: ManuallyDrop::new(root.inner.iter()),
            }
        } else {
            IterLevel {
                leaf: ManuallyDrop::new(root.leaf.iter()),
            }
        });
        FilterIter {
            filter,
            height,
            stack,
        }
    }

    pub(crate) fn empty(filter: Filter) -> FilterIter<'a, N, D, Key, Value, Filter> {
        FilterIter {
            filter,
            height: 0,
            stack: Vec::new(),
        }
    }
}

impl<'a, N, const D: usize, Key, Value, Filter> Iterator
    for FilterIter<'a, N, D, Key, Value, Filter>
where
    N: Ord,
    Key: Bounded<N, D>,
    Filter: SpatialFilter<N, D, Key>,
{
    type Item = &'a (Key, Value);

    fn next(&mut self) -> Option<Self::Item> {
        while !self.stack.is_empty() {
            let level = self.height - self.stack.len();
            if level == 0 {
                // Iterating over leaf node
                if let Some(value) = (*unsafe { &mut self.stack.last_mut().unwrap().leaf })
                    .find(|(key, _)| self.filter.test_key(key))
                {
                    return Some(value);
                } else {
                    unsafe { ManuallyDrop::drop(&mut self.stack.pop().unwrap().leaf) };
                }
            } else {
                // Iterating over inner node
                if let Some(entry) = (*unsafe { &mut self.stack.last_mut().unwrap().inner })
                    .find(|entry| self.filter.test_bounds(&entry.bounds))
                {
                    self.stack.push(if level > 1 {
                        IterLevel {
                            inner: ManuallyDrop::new((*unsafe { &entry.node.inner }).iter()),
                        }
                    } else {
                        IterLevel {
                            leaf: ManuallyDrop::new((*unsafe { &entry.node.leaf }).iter()),
                        }
                    })
                } else {
                    unsafe { ManuallyDrop::drop(&mut self.stack.pop().unwrap().inner) };
                }
            }
        }
        Option::None
    }
}
