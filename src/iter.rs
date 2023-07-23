use std::{cmp::min, mem::ManuallyDrop};

use crate::{bounds::Bounded, filter::SpatialFilter, fs_vec, node::Node};

union IterLevel<'a, N, const D: usize, Key, Value> {
    inner: ManuallyDrop<fs_vec::Iter<'a, Node<N, D, Key, Value>>>,
    leaf: ManuallyDrop<fs_vec::Iter<'a, (Key, Value)>>,
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
        stack.push(if height > 0 {
            IterLevel {
                inner: ManuallyDrop::new(fs_vec::Iter::new(&root.children)),
            }
        } else {
            IterLevel {
                leaf: ManuallyDrop::new(fs_vec::Iter::new(&root.children)),
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

impl<'a, N, const D: usize, Key, Value> Drop for Iter<'a, N, D, Key, Value> {
    fn drop(&mut self) {
        for i in 0..min(self.stack.len(), self.height) {
            unsafe {
                ManuallyDrop::drop(&mut self.stack[i].inner);
            }
        }
        if self.stack.len() == self.height + 1 {
            unsafe {
                ManuallyDrop::drop(&mut self.stack[self.height].leaf);
            }
        }
    }
}

impl<'a, N, const D: usize, Key, Value> Iterator for Iter<'a, N, D, Key, Value> {
    type Item = &'a (Key, Value);

    fn next(&mut self) -> Option<Self::Item> {
        while !self.stack.is_empty() {
            let level = (self.height + 1) - self.stack.len();
            if level == 0 {
                // Iterating over leaf node
                if let Some(entry) = (*unsafe { &mut self.stack.last_mut().unwrap().leaf }).next() {
                    return Some(entry);
                } else {
                    unsafe { ManuallyDrop::drop(&mut self.stack.pop().unwrap().leaf) };
                }
            } else {
                // Iterating over inner node
                if let Some(node) = (*unsafe { &mut self.stack.last_mut().unwrap().inner }).next() {
                    self.stack.push(if level > 1 {
                        IterLevel {
                            inner: ManuallyDrop::new(unsafe { fs_vec::Iter::new(&node.children) }),
                        }
                    } else {
                        IterLevel {
                            leaf: ManuallyDrop::new(unsafe { fs_vec::Iter::new(&node.children) }),
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
    ) -> FilterIter<'a, N, D, Key, Value, Filter>
    where
        Filter: SpatialFilter<N, D, Key>,
    {
        if !filter.test_bounds(&root.bounds) {
            return FilterIter {
                filter,
                height,
                stack: Vec::new(),
            };
        }
        let mut stack = Vec::with_capacity(height);
        stack.push(if height > 1 {
            IterLevel {
                inner: ManuallyDrop::new(fs_vec::Iter::new(&root.children)),
            }
        } else {
            IterLevel {
                leaf: ManuallyDrop::new(fs_vec::Iter::new(&root.children)),
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

impl<'a, N, const D: usize, Key, Value, Filter> Drop for FilterIter<'a, N, D, Key, Value, Filter> {
    fn drop(&mut self) {
        for i in 0..min(self.stack.len(), self.height) {
            unsafe {
                ManuallyDrop::drop(&mut self.stack[i].inner);
            }
        }
        if self.stack.len() == self.height + 1 {
            unsafe {
                ManuallyDrop::drop(&mut self.stack[self.height].leaf);
            }
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
            let level = (self.height + 1) - self.stack.len();
            if level == 0 {
                // Iterating over leaf node
                if let Some(entry) = (*unsafe { &mut self.stack.last_mut().unwrap().leaf })
                    .find(|(key, _)| self.filter.test_key(key))
                {
                    return Some(entry);
                } else {
                    unsafe { ManuallyDrop::drop(&mut self.stack.pop().unwrap().leaf) };
                }
            } else {
                // Iterating over inner node
                if let Some(node) = (*unsafe { &mut self.stack.last_mut().unwrap().inner })
                    .find(|node| self.filter.test_bounds(&node.bounds))
                {
                    self.stack.push(if level > 1 {
                        IterLevel {
                            inner: ManuallyDrop::new(unsafe { fs_vec::Iter::new(&node.children) }),
                        }
                    } else {
                        IterLevel {
                            leaf: ManuallyDrop::new(unsafe { fs_vec::Iter::new(&node.children) }),
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
