use std::{
    borrow::Borrow,
    cmp::{self, min, Reverse},
    collections::BinaryHeap,
    marker::PhantomData,
    mem::ManuallyDrop,
    slice,
};

use crate::{
    bounds::{Bounded, Bounds},
    filter::SpatialFilter,
    node::Node,
    ranking::Ranking,
};

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

pub struct FilterIter<'a, N, const D: usize, Key, Value, Q, Filter>
where
    Key: Borrow<Q>,
    Q: ?Sized,
    Filter: SpatialFilter<N, D, Q>,
{
    filter: Filter,
    // TODO: Use vector capacity instead of height?
    height: usize,
    stack: Vec<IterLevel<'a, N, D, Key, Value>>,

    _phantom: PhantomData<Q>,
}

impl<'a, N, const D: usize, Key, Value, Q, Filter> FilterIter<'a, N, D, Key, Value, Q, Filter>
where
    Key: Borrow<Q>,
    Q: ?Sized,
    Filter: SpatialFilter<N, D, Q>,
{
    pub(crate) unsafe fn new(
        height: usize,
        root: &'a Node<N, D, Key, Value>,
        filter: Filter,
    ) -> Self {
        if !filter.test_bounds(&root.bounds) {
            return FilterIter {
                filter,
                height,
                stack: Vec::new(),

                _phantom: PhantomData,
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

            _phantom: PhantomData,
        }
    }
}

impl<'a, N, const D: usize, Key, Value, Q, Filter> Drop
    for FilterIter<'a, N, D, Key, Value, Q, Filter>
where
    Key: Borrow<Q>,
    Q: ?Sized,
    Filter: SpatialFilter<N, D, Q>,
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

impl<'a, N, const D: usize, Key, Value, Q, Filter> Iterator
    for FilterIter<'a, N, D, Key, Value, Q, Filter>
where
    N: Ord,
    Key: Bounded<N, D> + Borrow<Q>,
    Q: ?Sized,
    Filter: SpatialFilter<N, D, Q>,
{
    type Item = (&'a Key, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        while !self.stack.is_empty() {
            let level = (self.height + 1) - self.stack.len();
            if level == 0 {
                // Iterating over leaf node
                if let Some(entry) = (*unsafe { &mut self.stack.last_mut().unwrap().leaf })
                    .find(|(key, _)| self.filter.test_key(key.borrow()))
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

pub struct FilterIterMut<'a, N, const D: usize, Key, Value, Q, Filter>
where
    Key: Borrow<Q>,
    Q: ?Sized,
    Filter: SpatialFilter<N, D, Q>,
{
    filter: Filter,
    height: usize,
    stack: Vec<IterLevelMut<'a, N, D, Key, Value>>,

    _phantom: PhantomData<Q>,
}

impl<'a, N, const D: usize, Key, Value, Q, Filter> FilterIterMut<'a, N, D, Key, Value, Q, Filter>
where
    Key: Borrow<Q>,
    Q: ?Sized,
    Filter: SpatialFilter<N, D, Q>,
{
    pub(crate) unsafe fn new(
        height: usize,
        root: &'a mut Node<N, D, Key, Value>,
        filter: Filter,
    ) -> Self {
        if !filter.test_bounds(&root.bounds) {
            return FilterIterMut {
                filter,
                height,
                stack: Vec::new(),

                _phantom: PhantomData,
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

            _phantom: PhantomData,
        }
    }
}

impl<'a, N, const D: usize, Key, Value, Q, Filter> Drop
    for FilterIterMut<'a, N, D, Key, Value, Q, Filter>
where
    Key: Borrow<Q>,
    Q: ?Sized,
    Filter: SpatialFilter<N, D, Q>,
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

impl<'a, N, const D: usize, Key, Value, Q, Filter> Iterator
    for FilterIterMut<'a, N, D, Key, Value, Q, Filter>
where
    Key: Borrow<Q>,
    Q: ?Sized,
    Filter: SpatialFilter<N, D, Q>,
{
    type Item = (&'a Key, &'a mut Value);

    fn next(&mut self) -> Option<Self::Item> {
        while !self.stack.is_empty() {
            let level = (self.height + 1) - self.stack.len();
            if level == 0 {
                // Iterating over leaf node
                if let Some(entry) = (*unsafe { &mut self.stack.last_mut().unwrap().leaf })
                    .find(|(key, _)| self.filter.test_key(key.borrow()))
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

struct Ranked<S, T> {
    score: S,
    value: T,
}

impl<S, T> PartialEq for Ranked<S, T>
where
    S: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl<S, T> Eq for Ranked<S, T> where S: Eq {}

impl<S, T> PartialOrd for Ranked<S, T>
where
    S: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl<S, T> Ord for Ranked<S, T>
where
    S: Ord,
{
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.score.cmp(&other.score)
    }
}

union SortedIterItem<'a, N, const D: usize, Key, Value> {
    node: &'a Node<N, D, Key, Value>,
    entry: &'a (Key, Value),
}

pub struct SortedIter<'a, N, const D: usize, Key, Value, R>
where
    R: Ranking<N, D, Key>,
{
    ranking: R,
    entries: BinaryHeap<Ranked<Reverse<(R::Metric, usize)>, SortedIterItem<'a, N, D, Key, Value>>>,

    _phantom: PhantomData<Bounds<N, D>>,
}

impl<'a, N, const D: usize, Key, Value, R> SortedIter<'a, N, D, Key, Value, R>
where
    R: Ranking<N, D, Key>,
{
    pub(crate) unsafe fn new(height: usize, root: &'a Node<N, D, Key, Value>, ranking: R) -> Self {
        let mut entries = BinaryHeap::new();
        entries.push(Ranked {
            score: Reverse((ranking.bounds_min(&root.bounds), height + 1)),
            value: SortedIterItem { node: root },
        });
        SortedIter {
            ranking,
            entries,

            _phantom: PhantomData,
        }
    }
}

impl<'a, N, const D: usize, Key, Value, R> Iterator for SortedIter<'a, N, D, Key, Value, R>
where
    R: Ranking<N, D, Key>,
{
    type Item = (&'a Key, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(Ranked {
            score: Reverse((_, level)),
            value,
        }) = self.entries.pop()
        {
            if level == 0 {
                let entry = unsafe { value.entry };
                return Some((&entry.0, &entry.1));
            } else {
                let node = unsafe { value.node };
                if level == 1 {
                    for entry in unsafe { node.leaf_children() } {
                        self.entries.push(Ranked {
                            score: Reverse((self.ranking.rank_key(&entry.0), level - 1)),
                            value: SortedIterItem { entry },
                        });
                    }
                } else {
                    for child in unsafe { node.inner_children() } {
                        self.entries.push(Ranked {
                            score: Reverse((self.ranking.bounds_min(&child.bounds), level - 1)),
                            value: SortedIterItem { node: child },
                        });
                    }
                }
            }
        }
        Option::None
    }
}

union SortedIterItemMut<'a, N, const D: usize, Key, Value> {
    node: &'a mut Node<N, D, Key, Value>,
    entry: &'a mut (Key, Value),
}

pub struct SortedIterMut<'a, N, const D: usize, Key, Value, R>
where
    R: Ranking<N, D, Key>,
{
    ranking: R,
    entries:
        BinaryHeap<Ranked<Reverse<(R::Metric, usize)>, SortedIterItemMut<'a, N, D, Key, Value>>>,

    _phantom: PhantomData<Bounds<N, D>>,
}

impl<'a, N, const D: usize, Key, Value, R> SortedIterMut<'a, N, D, Key, Value, R>
where
    R: Ranking<N, D, Key>,
{
    pub(crate) unsafe fn new(
        height: usize,
        root: &'a mut Node<N, D, Key, Value>,
        ranking: R,
    ) -> Self {
        let mut entries = BinaryHeap::new();
        entries.push(Ranked {
            score: Reverse((ranking.bounds_min(&root.bounds), height + 1)),
            value: SortedIterItemMut { node: root },
        });
        SortedIterMut {
            ranking,
            entries,

            _phantom: PhantomData,
        }
    }
}

impl<'a, N, const D: usize, Key, Value, R> Iterator for SortedIterMut<'a, N, D, Key, Value, R>
where
    R: Ranking<N, D, Key>,
{
    type Item = (&'a Key, &'a mut Value);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(Ranked {
            score: Reverse((_, level)),
            value,
        }) = self.entries.pop()
        {
            if level == 0 {
                let entry = unsafe { value.entry };
                return Some((&entry.0, &mut entry.1));
            } else {
                let node = unsafe { value.node };
                if level == 1 {
                    for entry in unsafe { node.leaf_children_mut() } {
                        self.entries.push(Ranked {
                            score: Reverse((self.ranking.rank_key(&entry.0), level - 1)),
                            value: SortedIterItemMut { entry },
                        });
                    }
                } else {
                    for child in unsafe { node.inner_children_mut() } {
                        self.entries.push(Ranked {
                            score: Reverse((self.ranking.bounds_min(&child.bounds), level - 1)),
                            value: SortedIterItemMut { node: child },
                        });
                    }
                }
            }
        }
        Option::None
    }
}

pub struct FilterSortedIter<'a, N, const D: usize, Key, Value, R, S>
where
    R: Ranking<N, D, Key, Metric = Option<S>>,
    S: Ord,
{
    ranking: R,
    entries: BinaryHeap<Ranked<Reverse<(S, usize)>, SortedIterItem<'a, N, D, Key, Value>>>,

    _phantom: PhantomData<Bounds<N, D>>,
}

impl<'a, N, const D: usize, Key, Value, R, S> FilterSortedIter<'a, N, D, Key, Value, R, S>
where
    R: Ranking<N, D, Key, Metric = Option<S>>,
    S: Ord,
{
    pub(crate) unsafe fn new(height: usize, root: &'a Node<N, D, Key, Value>, ranking: R) -> Self {
        let mut entries = BinaryHeap::new();
        if let Some(score) = ranking.bounds_min(&root.bounds) {
            entries.push(Ranked {
                score: Reverse((score, height + 1)),
                value: SortedIterItem { node: root },
            });
        }
        Self {
            ranking,
            entries,

            _phantom: PhantomData,
        }
    }
}

impl<'a, N, const D: usize, Key, Value, R, S> Iterator
    for FilterSortedIter<'a, N, D, Key, Value, R, S>
where
    R: Ranking<N, D, Key, Metric = Option<S>>,
    S: Ord,
{
    type Item = (&'a Key, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(Ranked {
            score: Reverse((_, level)),
            value,
        }) = self.entries.pop()
        {
            if level == 0 {
                let entry = unsafe { value.entry };
                return Some((&entry.0, &entry.1));
            } else {
                let node = unsafe { value.node };
                if level == 1 {
                    for entry in unsafe { node.leaf_children() } {
                        if let Some(score) = self.ranking.rank_key(&entry.0) {
                            self.entries.push(Ranked {
                                score: Reverse((score, level - 1)),
                                value: SortedIterItem { entry },
                            });
                        }
                    }
                } else {
                    for child in unsafe { node.inner_children() } {
                        if let Some(score) = self.ranking.bounds_min(&child.bounds) {
                            self.entries.push(Ranked {
                                score: Reverse((score, level - 1)),
                                value: SortedIterItem { node: child },
                            });
                        }
                    }
                }
            }
        }
        Option::None
    }
}

pub struct FilterSortedIterMut<'a, N, const D: usize, Key, Value, R, S>
where
    R: Ranking<N, D, Key, Metric = Option<S>>,
    S: Ord,
{
    ranking: R,
    entries: BinaryHeap<Ranked<Reverse<(S, usize)>, SortedIterItemMut<'a, N, D, Key, Value>>>,

    _phantom: PhantomData<Bounds<N, D>>,
}

impl<'a, N, const D: usize, Key, Value, R, S> FilterSortedIterMut<'a, N, D, Key, Value, R, S>
where
    R: Ranking<N, D, Key, Metric = Option<S>>,
    S: Ord,
{
    pub(crate) unsafe fn new(
        height: usize,
        root: &'a mut Node<N, D, Key, Value>,
        ranking: R,
    ) -> Self {
        let mut entries = BinaryHeap::new();
        if let Some(score) = ranking.bounds_min(&root.bounds) {
            entries.push(Ranked {
                score: Reverse((score, height + 1)),
                value: SortedIterItemMut { node: root },
            });
        }
        Self {
            ranking,
            entries,

            _phantom: PhantomData,
        }
    }
}

impl<'a, N, const D: usize, Key, Value, R, S> Iterator
    for FilterSortedIterMut<'a, N, D, Key, Value, R, S>
where
    R: Ranking<N, D, Key, Metric = Option<S>>,
    S: Ord,
{
    type Item = (&'a Key, &'a mut Value);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(Ranked {
            score: Reverse((_, level)),
            value,
        }) = self.entries.pop()
        {
            if level == 0 {
                let entry = unsafe { value.entry };
                return Some((&entry.0, &mut entry.1));
            } else {
                let node = unsafe { value.node };
                if level == 1 {
                    for entry in unsafe { node.leaf_children_mut() } {
                        if let Some(score) = self.ranking.rank_key(&entry.0) {
                            self.entries.push(Ranked {
                                score: Reverse((score, level - 1)),
                                value: SortedIterItemMut { entry },
                            });
                        }
                    }
                } else {
                    for child in unsafe { node.inner_children_mut() } {
                        if let Some(score) = self.ranking.bounds_min(&child.bounds) {
                            self.entries.push(Ranked {
                                score: Reverse((score, level - 1)),
                                value: SortedIterItemMut { node: child },
                            });
                        }
                    }
                }
            }
        }
        Option::None
    }
}
