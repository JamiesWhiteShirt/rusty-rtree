use std::{
    borrow::Borrow,
    cmp::{self, Reverse},
    collections::BinaryHeap,
    marker::PhantomData,
    num::NonZeroUsize,
    ops::DerefMut,
    slice,
};

use crate::{
    bounds::Bounds,
    filter::{NoFilter, SpatialFilter},
    iter_stack::IterStack,
    node::Node,
    ranking::Ranking,
    util::empty_slice,
};

pub(crate) type TreeIter<'a, N, const D: usize, Key, Value> =
    IterStack<slice::Iter<'a, (Key, Value)>, slice::Iter<'a, Node<N, D, Key, Value>>>;

pub struct Iter<Stack> {
    stack: Stack,
}

impl<'a, N, const D: usize, Key, Value, Stack> Iter<Stack>
where
    Stack: DerefMut<Target = TreeIter<'a, N, D, Key, Value>>,
    Node<N, D, Key, Value>: 'a,
{
    pub(crate) unsafe fn new_from_stack(stack: Stack) -> Self {
        Self { stack }
    }
}

impl<'a, N, const D: usize, Key, Value> Iter<Box<TreeIter<'a, N, D, Key, Value>>> {
    pub(crate) fn new_empty() -> Self {
        Self {
            stack: IterStack::empty_box(),
        }
    }

    pub(crate) unsafe fn new(height: usize, root: &'a Node<N, D, Key, Value>) -> Self {
        // Initialize the stack with empty iterators for each level of the tree
        // except the root. In the first iteration, all iterators but the root
        // will act as if they are exhausted, so the stack will be initialized
        // with true iterators starting from the root.
        let mut stack = IterStack::new_box(
            empty_slice().iter(),
            (0..height).map(|_| empty_slice().iter()),
        );
        if let Some(height) = NonZeroUsize::new(height) {
            stack[height] = root.inner_children().iter();
        } else {
            *stack.leaf_mut() = root.leaf_children().iter();
        }
        Self { stack }
    }
}

impl<'a, N, const D: usize, Key, Value, Stack> Iter<Stack>
where
    Stack: DerefMut<Target = TreeIter<'a, N, D, Key, Value>>,
    Node<N, D, Key, Value>: 'a,
{
    fn next<Q>(&mut self, filter: &mut impl SpatialFilter<N, D, Q>) -> Option<(&'a Key, &'a Value)>
    where
        Key: Borrow<Q>,
        Q: ?Sized,
    {
        let mut level = 0;
        while level < self.stack.len() {
            if let Some(nz_level) = NonZeroUsize::new(level) {
                if let Some(node) =
                    self.stack[nz_level].find(|node| filter.test_bounds(&node.bounds))
                {
                    level -= 1;
                    if let Some(nz_level) = NonZeroUsize::new(level) {
                        // SAFETY: The level is non-zero, so the node must be an inner node.
                        self.stack[nz_level] = unsafe { node.inner_children() }.iter();
                    } else {
                        // SAFETY: The level is zero, so the node must be a leaf node.
                        *self.stack.leaf_mut() = unsafe { node.leaf_children() }.iter();
                    }
                } else {
                    level += 1;
                }
            } else {
                if let Some(entry) = self
                    .stack
                    .leaf_mut()
                    .find(|(key, _)| filter.test_key(key.borrow()))
                {
                    return Some((&entry.0, &entry.1));
                } else {
                    level += 1;
                }
            }
        }
        None
    }
}

impl<'a, N, const D: usize, Key, Value, Stack> Iterator for Iter<Stack>
where
    Stack: DerefMut<Target = TreeIter<'a, N, D, Key, Value>>,
    Node<N, D, Key, Value>: 'a,
{
    type Item = (&'a Key, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        self.next(&mut NoFilter)
    }
}

pub struct FilterIter<Stack, Q, F>
where
    Q: ?Sized,
{
    iter: Iter<Stack>,
    filter: F,
    _phantom: PhantomData<Q>,
}

impl<'a, N, const D: usize, Key, Value, Q, F> FilterIter<Box<TreeIter<'a, N, D, Key, Value>>, Q, F>
where
    Key: Borrow<Q>,
    Q: ?Sized,
    F: SpatialFilter<N, D, Q>,
{
    pub(crate) unsafe fn new(height: usize, root: &'a Node<N, D, Key, Value>, filter: F) -> Self {
        Self {
            iter: if filter.test_bounds(&root.borrow().bounds) {
                Iter::new(height, root)
            } else {
                Iter::new_empty()
            },
            filter,
            _phantom: PhantomData,
        }
    }
}

impl<'a, N, const D: usize, Key, Value, Stack, Q, F> FilterIter<Stack, Q, F>
where
    Stack: DerefMut<Target = TreeIter<'a, N, D, Key, Value>>,
    Node<N, D, Key, Value>: 'a,
    Q: ?Sized,
{
    pub(crate) unsafe fn new_from_stack(stack: Stack, filter: F) -> Self {
        Self {
            iter: Iter::new_from_stack(stack),
            filter,
            _phantom: PhantomData,
        }
    }
}

impl<'a, N, const D: usize, Key, Value, Stack, Q, F> Iterator for FilterIter<Stack, Q, F>
where
    Stack: DerefMut<Target = TreeIter<'a, N, D, Key, Value>>,
    Node<N, D, Key, Value>: 'a,
    Key: Borrow<Q>,
    Q: ?Sized,
    F: SpatialFilter<N, D, Q>,
{
    type Item = (&'a Key, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next(&mut self.filter)
    }
}

pub(crate) type TreeIterMut<'a, N, const D: usize, Key, Value> =
    IterStack<slice::IterMut<'a, (Key, Value)>, slice::IterMut<'a, Node<N, D, Key, Value>>>;

pub struct IterMut<Stack> {
    stack: Stack,
}

impl<'a, N, const D: usize, Key, Value> IterMut<Box<TreeIterMut<'a, N, D, Key, Value>>> {
    pub(crate) fn new_empty() -> Self {
        Self {
            stack: IterStack::empty_box(),
        }
    }

    pub(crate) unsafe fn new(height: usize, root: &'a mut Node<N, D, Key, Value>) -> Self {
        // Initialize the stack with empty iterators for each level of the tree
        // except the root. In the first iteration, all iterators but the root
        // will act as if they are exhausted, so the stack will be initialized
        // with true iterators starting from the root.
        let mut stack = IterStack::new_box(
            empty_slice().iter_mut(),
            (0..height).map(|_| empty_slice().iter_mut()),
        );
        if let Some(height) = NonZeroUsize::new(height) {
            stack[height] = root.inner_children_mut().iter_mut();
        } else {
            *stack.leaf_mut() = root.leaf_children_mut().iter_mut();
        }
        Self { stack }
    }
}

impl<'a, N, const D: usize, Key, Value, Stack> IterMut<Stack>
where
    Stack: DerefMut<Target = TreeIterMut<'a, N, D, Key, Value>>,
    Node<N, D, Key, Value>: 'a,
{
    fn next<Q>(
        &mut self,
        filter: &mut impl SpatialFilter<N, D, Q>,
    ) -> Option<(&'a Key, &'a mut Value)>
    where
        Key: Borrow<Q>,
        Q: ?Sized,
    {
        let mut level = 0;
        while level < self.stack.len() {
            if let Some(nz_level) = NonZeroUsize::new(level) {
                if let Some(node) =
                    self.stack[nz_level].find(|node| filter.test_bounds(&node.bounds))
                {
                    level -= 1;
                    if let Some(nz_level) = NonZeroUsize::new(level) {
                        // SAFETY: The level is non-zero, so the node must be an inner node.
                        self.stack[nz_level] = unsafe { node.inner_children_mut() }.iter_mut();
                    } else {
                        // SAFETY: The level is zero, so the node must be a leaf node.
                        *self.stack.leaf_mut() = unsafe { node.leaf_children_mut() }.iter_mut();
                    }
                } else {
                    level += 1;
                }
            } else {
                if let Some(entry) = self
                    .stack
                    .leaf_mut()
                    .find(|(key, _)| filter.test_key(key.borrow()))
                {
                    return Some((&entry.0, &mut entry.1));
                } else {
                    level += 1;
                }
            }
        }
        None
    }
}

impl<'a, N, const D: usize, Key, Value, Stack> Iterator for IterMut<Stack>
where
    Stack: DerefMut<Target = TreeIterMut<'a, N, D, Key, Value>>,
    Node<N, D, Key, Value>: 'a,
{
    type Item = (&'a Key, &'a mut Value);

    fn next(&mut self) -> Option<Self::Item> {
        self.next(&mut NoFilter)
    }
}

pub struct FilterIterMut<Stack, Q, F>
where
    Q: ?Sized,
{
    iter: IterMut<Stack>,
    filter: F,
    _phantom: PhantomData<Q>,
}

impl<'a, N, const D: usize, Key, Value, Q, F>
    FilterIterMut<Box<TreeIterMut<'a, N, D, Key, Value>>, Q, F>
where
    Key: Borrow<Q>,
    Q: ?Sized,
    F: SpatialFilter<N, D, Q>,
{
    pub(crate) unsafe fn new(
        height: usize,
        root: &'a mut Node<N, D, Key, Value>,
        filter: F,
    ) -> Self {
        Self {
            iter: if filter.test_bounds(&root.bounds) {
                IterMut::new(height, root)
            } else {
                IterMut::new_empty()
            },
            filter,
            _phantom: PhantomData,
        }
    }
}

impl<'a, N, const D: usize, Key, Value, Stack, Q, F> Iterator for FilterIterMut<Stack, Q, F>
where
    Stack: DerefMut<Target = TreeIterMut<'a, N, D, Key, Value>>,
    Node<N, D, Key, Value>: 'a,
    Key: Borrow<Q>,
    Q: ?Sized,
    F: SpatialFilter<N, D, Q>,
{
    type Item = (&'a Key, &'a mut Value);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next(&mut self.filter)
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
