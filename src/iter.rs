use std::{cmp::min, mem::ManuallyDrop, slice};

use crate::{bounds::Bounded, filter::SpatialFilter, node::Node};

union IterLevel<'a, N, const D: usize, Key, Value> {
    inner: ManuallyDrop<slice::Iter<'a, Node<N, D, Key, Value>>>,
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
        stack.push(if height > 0 {
            IterLevel {
                inner: ManuallyDrop::new(root.inner_children().iter()),
            }
        } else {
            IterLevel {
                leaf: ManuallyDrop::new(root.leaf_children().iter()),
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
        let stack_len = self.stack.len();
        for iter in &mut self.stack[0..min(stack_len, self.height)] {
            unsafe {
                ManuallyDrop::drop(&mut iter.inner);
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
    type Item = (&'a Key, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        while !self.stack.is_empty() {
            let level = (self.height + 1) - self.stack.len();
            if level == 0 {
                // Iterating over leaf node
                if let Some(entry) = (*unsafe { &mut self.stack.last_mut().unwrap().leaf }).next() {
                    return Some((&entry.0, &entry.1));
                } else {
                    unsafe { ManuallyDrop::drop(&mut self.stack.pop().unwrap().leaf) };
                }
            } else {
                // Iterating over inner node
                if let Some(node) = (*unsafe { &mut self.stack.last_mut().unwrap().inner }).next() {
                    self.stack.push(if level > 1 {
                        IterLevel {
                            inner: ManuallyDrop::new(unsafe { node.inner_children() }.iter()),
                        }
                    } else {
                        IterLevel {
                            leaf: ManuallyDrop::new(unsafe { node.leaf_children() }.iter()),
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

union IterLevelMut<'a, N, const D: usize, Key, Value> {
    inner: ManuallyDrop<slice::IterMut<'a, Node<N, D, Key, Value>>>,
    leaf: ManuallyDrop<slice::IterMut<'a, (Key, Value)>>,
}

pub struct IterMut<'a, N, const D: usize, Key, Value> {
    height: usize,
    stack: Vec<IterLevelMut<'a, N, D, Key, Value>>,
}

impl<'a, N, const D: usize, Key, Value> IterMut<'a, N, D, Key, Value> {
    pub(crate) unsafe fn new(
        height: usize,
        root: &'a mut Node<N, D, Key, Value>,
    ) -> IterMut<'a, N, D, Key, Value> {
        let mut stack = Vec::with_capacity(height);
        stack.push(if height > 0 {
            IterLevelMut {
                inner: ManuallyDrop::new(root.inner_children_mut().iter_mut()),
            }
        } else {
            IterLevelMut {
                leaf: ManuallyDrop::new(root.leaf_children_mut().iter_mut()),
            }
        });
        IterMut { height, stack }
    }

    pub(crate) fn empty() -> IterMut<'a, N, D, Key, Value> {
        IterMut {
            height: 0,
            stack: Vec::new(),
        }
    }
}

impl<'a, N, const D: usize, Key, Value> Drop for IterMut<'a, N, D, Key, Value> {
    fn drop(&mut self) {
        let stack_len = self.stack.len();
        for iter in &mut self.stack[0..min(stack_len, self.height)] {
            unsafe {
                ManuallyDrop::drop(&mut iter.inner);
            }
        }
        if stack_len == self.height + 1 {
            unsafe {
                ManuallyDrop::drop(&mut self.stack[self.height].leaf);
            }
        }
    }
}

impl<'a, N, const D: usize, Key, Value> Iterator for IterMut<'a, N, D, Key, Value> {
    type Item = (&'a Key, &'a mut Value);

    fn next(&mut self) -> Option<Self::Item> {
        while !self.stack.is_empty() {
            let level = (self.height + 1) - self.stack.len();
            if level == 0 {
                // Iterating over leaf node
                if let Some(entry) = (*unsafe { &mut self.stack.last_mut().unwrap().leaf }).next() {
                    return Some((&entry.0, &mut entry.1));
                } else {
                    unsafe { ManuallyDrop::drop(&mut self.stack.pop().unwrap().leaf) };
                }
            } else {
                // Iterating over inner node
                if let Some(node) = (*unsafe { &mut self.stack.last_mut().unwrap().inner }).next() {
                    self.stack.push(if level > 1 {
                        IterLevelMut {
                            inner: ManuallyDrop::new(
                                unsafe { node.inner_children_mut() }.iter_mut(),
                            ),
                        }
                    } else {
                        IterLevelMut {
                            leaf: ManuallyDrop::new(unsafe { node.leaf_children_mut() }.iter_mut()),
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

pub struct FilterIter<'a, N, const D: usize, Key, Value, Filter>
where
    Filter: SpatialFilter<N, D, Key>,
{
    filter: Filter,
    // TODO: Use vector capacity instead of height?
    height: usize,
    stack: Vec<IterLevel<'a, N, D, Key, Value>>,
}

impl<'a, N, const D: usize, Key, Value, Filter> FilterIter<'a, N, D, Key, Value, Filter>
where
    Filter: SpatialFilter<N, D, Key>,
{
    pub(crate) unsafe fn new(
        height: usize,
        root: &'a Node<N, D, Key, Value>,
        filter: Filter,
    ) -> FilterIter<'a, N, D, Key, Value, Filter> {
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
                inner: ManuallyDrop::new(root.inner_children().iter()),
            }
        } else {
            IterLevel {
                leaf: ManuallyDrop::new(root.leaf_children().iter()),
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

impl<'a, N, const D: usize, Key, Value, Filter> Drop for FilterIter<'a, N, D, Key, Value, Filter>
where
    Filter: SpatialFilter<N, D, Key>,
{
    fn drop(&mut self) {
        let stack_len = self.stack.len();
        for iter in &mut self.stack[0..min(stack_len, self.height)] {
            unsafe {
                ManuallyDrop::drop(&mut iter.inner);
            }
        }
        if stack_len == self.height + 1 {
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
    type Item = (&'a Key, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        while !self.stack.is_empty() {
            let level = (self.height + 1) - self.stack.len();
            if level == 0 {
                // Iterating over leaf node
                if let Some(entry) = (*unsafe { &mut self.stack.last_mut().unwrap().leaf })
                    .find(|(key, _)| self.filter.test_key(key))
                {
                    return Some((&entry.0, &entry.1));
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
                            inner: ManuallyDrop::new(unsafe { node.inner_children() }.iter()),
                        }
                    } else {
                        IterLevel {
                            leaf: ManuallyDrop::new(unsafe { node.leaf_children() }.iter()),
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

pub struct FilterIterMut<'a, N, const D: usize, Key, Value, Filter>
where
    Filter: SpatialFilter<N, D, Key>,
{
    filter: Filter,
    height: usize,
    stack: Vec<IterLevelMut<'a, N, D, Key, Value>>,
}

impl<'a, N, const D: usize, Key, Value, Filter> FilterIterMut<'a, N, D, Key, Value, Filter>
where
    Filter: SpatialFilter<N, D, Key>,
{
    pub(crate) unsafe fn new(
        height: usize,
        root: &'a mut Node<N, D, Key, Value>,
        filter: Filter,
    ) -> FilterIterMut<'a, N, D, Key, Value, Filter> {
        if !filter.test_bounds(&root.bounds) {
            return FilterIterMut {
                filter,
                height,
                stack: Vec::new(),
            };
        }
        let mut stack = Vec::with_capacity(height);
        stack.push(if height > 1 {
            IterLevelMut {
                inner: ManuallyDrop::new(root.inner_children_mut().iter_mut()),
            }
        } else {
            IterLevelMut {
                leaf: ManuallyDrop::new(root.leaf_children_mut().iter_mut()),
            }
        });
        FilterIterMut {
            filter,
            height,
            stack,
        }
    }

    pub(crate) fn empty(filter: Filter) -> FilterIterMut<'a, N, D, Key, Value, Filter> {
        FilterIterMut {
            filter,
            height: 0,
            stack: Vec::new(),
        }
    }
}

impl<'a, N, const D: usize, Key, Value, Filter> Drop for FilterIterMut<'a, N, D, Key, Value, Filter>
where
    Filter: SpatialFilter<N, D, Key>,
{
    fn drop(&mut self) {
        let stack_len = self.stack.len();
        for iter in &mut self.stack[0..min(stack_len, self.height)] {
            unsafe {
                ManuallyDrop::drop(&mut iter.inner);
            }
        }
        if stack_len == self.height + 1 {
            unsafe {
                ManuallyDrop::drop(&mut self.stack[self.height].leaf);
            }
        }
    }
}

impl<'a, N, const D: usize, Key, Value, Filter> Iterator
    for FilterIterMut<'a, N, D, Key, Value, Filter>
where
    Filter: SpatialFilter<N, D, Key>,
{
    type Item = (&'a Key, &'a mut Value);

    fn next(&mut self) -> Option<Self::Item> {
        while !self.stack.is_empty() {
            let level = (self.height + 1) - self.stack.len();
            if level == 0 {
                // Iterating over leaf node
                if let Some(entry) = (*unsafe { &mut self.stack.last_mut().unwrap().leaf })
                    .find(|(key, _)| self.filter.test_key(key))
                {
                    return Some((&entry.0, &mut entry.1));
                } else {
                    unsafe { ManuallyDrop::drop(&mut self.stack.pop().unwrap().leaf) };
                }
            } else {
                // Iterating over inner node
                if let Some(node) = (*unsafe { &mut self.stack.last_mut().unwrap().inner })
                    .find(|node| self.filter.test_bounds(&node.bounds))
                {
                    self.stack.push(if level > 1 {
                        IterLevelMut {
                            inner: ManuallyDrop::new(
                                unsafe { node.inner_children_mut() }.iter_mut(),
                            ),
                        }
                    } else {
                        IterLevelMut {
                            leaf: ManuallyDrop::new(unsafe { node.leaf_children_mut() }.iter_mut()),
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
