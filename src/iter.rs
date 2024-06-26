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
    filter::{NoFilter, SpatialFilter},
    iter_stack::IterStack,
    node::{Node, NodeChildrenRef, NodeChildrenRefMut, NodeRef, NodeRefMut},
    ranking::Ranking,
    util::empty_slice,
};

pub(crate) type TreeIter<'a, B, Key, Value> =
    IterStack<slice::Iter<'a, (Key, Value)>, slice::Iter<'a, Node<B, Key, Value>>>;

pub struct Iter<Stack> {
    stack: Stack,
}

impl<'a, B, Key, Value, Stack> Iter<Stack>
where
    Stack: DerefMut<Target = TreeIter<'a, B, Key, Value>>,
    Node<B, Key, Value>: 'a,
{
    pub(crate) unsafe fn new_from_stack(stack: Stack) -> Self {
        Self { stack }
    }
}

impl<'a, B, Key, Value> Iter<Box<TreeIter<'a, B, Key, Value>>> {
    pub(crate) fn new_empty() -> Self {
        Self {
            stack: IterStack::empty_box(),
        }
    }

    pub(crate) fn new(root: NodeRef<'a, B, Key, Value>) -> Self {
        // Initialize the stack with empty iterators for each level of the tree
        // except the root. In the first iteration, all iterators but the root
        // will act as if they are exhausted, so the stack will be initialized
        // with true iterators starting from the root.
        let mut stack = IterStack::new_box(
            empty_slice().iter(),
            (0..root.level()).map(|_| empty_slice().iter()),
        );
        match root.children() {
            NodeChildrenRef::Inner(children) => {
                stack[children.level()] = children.vec().iter();
            }
            NodeChildrenRef::Leaf(children) => {
                *stack.leaf_mut() = children.iter();
            }
        }
        Self { stack }
    }
}

impl<'a, B, Key, Value, Stack> Iter<Stack>
where
    Stack: DerefMut<Target = TreeIter<'a, B, Key, Value>>,
    Node<B, Key, Value>: 'a,
{
    fn next<Q>(&mut self, filter: &mut impl SpatialFilter<B, Q>) -> Option<(&'a Key, &'a Value)>
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
            } else if let Some(entry) = self
                .stack
                .leaf_mut()
                .find(|(key, _)| filter.test_key(key.borrow()))
            {
                return Some((&entry.0, &entry.1));
            } else {
                level += 1;
            }
        }
        None
    }
}

impl<'a, B, Key, Value, Stack> Iterator for Iter<Stack>
where
    Stack: DerefMut<Target = TreeIter<'a, B, Key, Value>>,
    Node<B, Key, Value>: 'a,
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

impl<'a, B, Key, Value, Q, F> FilterIter<Box<TreeIter<'a, B, Key, Value>>, Q, F>
where
    Key: Borrow<Q>,
    Q: ?Sized,
    F: SpatialFilter<B, Q>,
{
    pub(crate) fn new(root: NodeRef<'a, B, Key, Value>, filter: F) -> Self {
        Self {
            iter: if filter.test_bounds(&root.node().bounds) {
                Iter::new(root)
            } else {
                Iter::new_empty()
            },
            filter,
            _phantom: PhantomData,
        }
    }
}

impl<'a, B, Key, Value, Stack, Q, F> FilterIter<Stack, Q, F>
where
    Stack: DerefMut<Target = TreeIter<'a, B, Key, Value>>,
    Node<B, Key, Value>: 'a,
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

impl<'a, B, Key, Value, Stack, Q, F> Iterator for FilterIter<Stack, Q, F>
where
    Stack: DerefMut<Target = TreeIter<'a, B, Key, Value>>,
    Node<B, Key, Value>: 'a,
    Key: Borrow<Q>,
    Q: ?Sized,
    F: SpatialFilter<B, Q>,
{
    type Item = (&'a Key, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next(&mut self.filter)
    }
}

pub(crate) type TreeIterMut<'a, B, Key, Value> =
    IterStack<slice::IterMut<'a, (Key, Value)>, slice::IterMut<'a, Node<B, Key, Value>>>;

pub struct IterMut<Stack> {
    stack: Stack,
}

impl<'a, B, Key, Value> IterMut<Box<TreeIterMut<'a, B, Key, Value>>> {
    pub(crate) fn new_empty() -> Self {
        Self {
            stack: IterStack::empty_box(),
        }
    }

    pub(crate) fn new(root: NodeRefMut<'a, B, Key, Value>) -> Self {
        // Initialize the stack with empty iterators for each level of the tree
        // except the root. In the first iteration, all iterators but the root
        // will act as if they are exhausted, so the stack will be initialized
        // with true iterators starting from the root.
        let mut stack = IterStack::new_box(
            empty_slice().iter_mut(),
            (0..root.level()).map(|_| empty_slice().iter_mut()),
        );
        match root.into_children_mut() {
            NodeChildrenRefMut::Inner(children) => {
                let level = children.level();
                stack[level] = children.into_vec_mut().into_iter_mut();
            }
            NodeChildrenRefMut::Leaf(children) => {
                *stack.leaf_mut() = children.into_iter_mut();
            }
        }
        Self { stack }
    }
}

impl<'a, B, Key, Value, Stack> IterMut<Stack>
where
    Stack: DerefMut<Target = TreeIterMut<'a, B, Key, Value>>,
    Node<B, Key, Value>: 'a,
{
    fn next<Q>(&mut self, filter: &mut impl SpatialFilter<B, Q>) -> Option<(&'a Key, &'a mut Value)>
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
            } else if let Some(entry) = self
                .stack
                .leaf_mut()
                .find(|(key, _)| filter.test_key(key.borrow()))
            {
                return Some((&entry.0, &mut entry.1));
            } else {
                level += 1;
            }
        }
        None
    }
}

impl<'a, B, Key, Value, Stack> Iterator for IterMut<Stack>
where
    Stack: DerefMut<Target = TreeIterMut<'a, B, Key, Value>>,
    Node<B, Key, Value>: 'a,
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

impl<'a, B, Key, Value, Q, F> FilterIterMut<Box<TreeIterMut<'a, B, Key, Value>>, Q, F>
where
    Key: Borrow<Q>,
    Q: ?Sized,
    F: SpatialFilter<B, Q>,
{
    pub(crate) fn new(root: NodeRefMut<'a, B, Key, Value>, filter: F) -> Self {
        Self {
            iter: if filter.test_bounds(&root.node().bounds) {
                IterMut::new(root)
            } else {
                IterMut::new_empty()
            },
            filter,
            _phantom: PhantomData,
        }
    }
}

impl<'a, B, Key, Value, Stack, Q, F> Iterator for FilterIterMut<Stack, Q, F>
where
    Stack: DerefMut<Target = TreeIterMut<'a, B, Key, Value>>,
    Node<B, Key, Value>: 'a,
    Key: Borrow<Q>,
    Q: ?Sized,
    F: SpatialFilter<B, Q>,
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

union SortedIterItem<'a, B, Key, Value> {
    node: &'a Node<B, Key, Value>,
    entry: &'a (Key, Value),
}

type IterHeap<Metric, Item> = BinaryHeap<Ranked<Reverse<(Metric, usize)>, Item>>;

pub struct SortedIter<'a, B, Key, Value, R>
where
    R: Ranking<B, Key>,
{
    ranking: R,
    entries: IterHeap<R::Metric, SortedIterItem<'a, B, Key, Value>>,

    _phantom: PhantomData<B>,
}

impl<'a, B, Key, Value, R> SortedIter<'a, B, Key, Value, R>
where
    R: Ranking<B, Key>,
{
    pub(crate) fn new(root: NodeRef<'a, B, Key, Value>, ranking: R) -> Self {
        let mut entries = BinaryHeap::new();
        entries.push(Ranked {
            score: Reverse((ranking.bounds_min(&root.node().bounds), root.level() + 1)),
            value: SortedIterItem { node: root.node() },
        });
        SortedIter {
            ranking,
            entries,

            _phantom: PhantomData,
        }
    }
}

impl<'a, B, Key, Value, R> Iterator for SortedIter<'a, B, Key, Value, R>
where
    R: Ranking<B, Key>,
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

union SortedIterItemMut<'a, B, Key, Value> {
    node: &'a mut Node<B, Key, Value>,
    entry: &'a mut (Key, Value),
}

pub struct SortedIterMut<'a, B, Key, Value, R>
where
    R: Ranking<B, Key>,
{
    ranking: R,
    entries: IterHeap<R::Metric, SortedIterItemMut<'a, B, Key, Value>>,

    _phantom: PhantomData<B>,
}

impl<'a, B, Key, Value, R> SortedIterMut<'a, B, Key, Value, R>
where
    R: Ranking<B, Key>,
{
    pub(crate) fn new(root: NodeRefMut<'a, B, Key, Value>, ranking: R) -> Self {
        let mut entries = BinaryHeap::new();
        entries.push(Ranked {
            score: Reverse((ranking.bounds_min(&root.node().bounds), root.level() + 1)),
            value: SortedIterItemMut {
                node: root.into_node(),
            },
        });
        SortedIterMut {
            ranking,
            entries,

            _phantom: PhantomData,
        }
    }
}

impl<'a, B, Key, Value, R> Iterator for SortedIterMut<'a, B, Key, Value, R>
where
    R: Ranking<B, Key>,
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

pub struct FilterSortedIter<'a, B, Key, Value, R, S>
where
    R: Ranking<B, Key, Metric = Option<S>>,
    S: Ord,
{
    ranking: R,
    entries: IterHeap<S, SortedIterItem<'a, B, Key, Value>>,

    _phantom: PhantomData<B>,
}

impl<'a, B, Key, Value, R, S> FilterSortedIter<'a, B, Key, Value, R, S>
where
    R: Ranking<B, Key, Metric = Option<S>>,
    S: Ord,
{
    pub(crate) fn new(root: NodeRef<'a, B, Key, Value>, ranking: R) -> Self {
        let mut entries = BinaryHeap::new();
        if let Some(score) = ranking.bounds_min(&root.node().bounds) {
            entries.push(Ranked {
                score: Reverse((score, root.level() + 1)),
                value: SortedIterItem { node: root.node() },
            });
        }
        Self {
            ranking,
            entries,

            _phantom: PhantomData,
        }
    }
}

impl<'a, B, Key, Value, R, S> Iterator for FilterSortedIter<'a, B, Key, Value, R, S>
where
    R: Ranking<B, Key, Metric = Option<S>>,
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

pub struct FilterSortedIterMut<'a, B, Key, Value, R, S>
where
    R: Ranking<B, Key, Metric = Option<S>>,
    S: Ord,
{
    ranking: R,
    entries: IterHeap<S, SortedIterItemMut<'a, B, Key, Value>>,

    _phantom: PhantomData<B>,
}

impl<'a, B, Key, Value, R, S> FilterSortedIterMut<'a, B, Key, Value, R, S>
where
    R: Ranking<B, Key, Metric = Option<S>>,
    S: Ord,
{
    pub(crate) fn new(root: NodeRefMut<'a, B, Key, Value>, ranking: R) -> Self {
        let mut entries = BinaryHeap::new();
        if let Some(score) = ranking.bounds_min(&root.node().bounds) {
            entries.push(Ranked {
                score: Reverse((score, root.level() + 1)),
                value: SortedIterItemMut {
                    node: root.into_node(),
                },
            });
        }
        Self {
            ranking,
            entries,

            _phantom: PhantomData,
        }
    }
}

impl<'a, B, Key, Value, R, S> Iterator for FilterSortedIterMut<'a, B, Key, Value, R, S>
where
    R: Ranking<B, Key, Metric = Option<S>>,
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
