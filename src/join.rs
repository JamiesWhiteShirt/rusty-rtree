use std::{borrow::Borrow, cmp, marker::PhantomData, num::NonZeroUsize, slice};

use crate::{filter::JoinFilter, iter_stack::IterStack, node::Node, util::empty_slice};

struct RewindableIter<'a, T> {
    slice: &'a [T],
    i: usize,
}

impl<'a, T> RewindableIter<'a, T> {
    fn new(slice: &'a [T]) -> Self {
        Self { slice, i: 0 }
    }

    fn get(&self) -> Option<&'a T> {
        if self.i < self.slice.len() {
            Some(&self.slice[self.i])
        } else {
            None
        }
    }

    fn advance(&mut self) -> bool {
        self.i += 1;
        if self.i >= self.slice.len() {
            self.i = 0;
            true
        } else {
            false
        }
    }
}

struct ProductIter<'a, 'b, T0, T1>(RewindableIter<'a, T0>, &'b [T1]);

impl<'a, 'b, T0, T1> ProductIter<'a, 'b, T0, T1> {
    fn new(slice0: &'a [T0], slice1: &'b [T1]) -> Self {
        Self(RewindableIter::new(slice0), slice1)
    }
}

impl<'a, 'b, T0, T1> Iterator for ProductIter<'a, 'b, T0, T1> {
    type Item = (&'a T0, &'b T1);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item0) = self.0.get() {
            if let Some(item1) = self.1.first() {
                if self.0.advance() {
                    self.1 = &self.1[1..];
                }
                return Some((item0, item1));
            }
        }
        None
    }
}

/// The iterator will simultaneously descend into both trees, joining the children of the nodes
/// using the provided filter until it reaches the leaf nodes. In case the trees have different
/// heights, the iterator will simultaneously descend into both trees until it reaches the leaf
/// nodes of the shorter tree, and then join the leaf nodes of the shorter tree with the inner nodes
/// of the taller tree.
pub struct JoinIter<
    'a,
    'b,
    N0,
    N1,
    const D0: usize,
    const D1: usize,
    Key0,
    Key1,
    Value0,
    Value1,
    Q0,
    Q1,
    Filter,
> where
    Q0: ?Sized,
    Q1: ?Sized,
{
    stack: Box<
        IterStack<
            ProductIter<'a, 'b, (Key0, Value0), (Key1, Value1)>,
            ProductIter<'a, 'b, Node<N0, D0, Key0, Value0>, Node<N1, D1, Key1, Value1>>,
        >,
    >,
    padding0: usize,
    padding1: usize,
    filter: Filter,
    _phantom: PhantomData<(&'a Q0, &'b Q1)>,
}

impl<
        'a,
        'b,
        N0,
        N1,
        const D0: usize,
        const D1: usize,
        Key0,
        Key1,
        Value0,
        Value1,
        Q0,
        Q1,
        Filter,
    > JoinIter<'a, 'b, N0, N1, D0, D1, Key0, Key1, Value0, Value1, Q0, Q1, Filter>
where
    Key0: Borrow<Q0>,
    Key1: Borrow<Q1>,
    Q0: ?Sized,
    Q1: ?Sized,
    Filter: JoinFilter<N0, N1, D0, D1, Q0, Q1>,
{
    /// Adds an iterator to the stack to join the nodes. If both nodes are of the same type, the
    /// iterator will join the children of both nodes. Otherwise, the iterator will join the
    /// children of the inner node with the leaf node.
    ///
    /// The iterator is added to the stack at the greater of the two levels.
    ///
    /// # Safety
    ///
    /// The levels must be valid for the given nodes.
    unsafe fn start_join(
        &mut self,
        node0: &'a Node<N0, D0, Key0, Value0>,
        level0: usize,
        node1: &'b Node<N1, D1, Key1, Value1>,
        level1: usize,
    ) {
        if let Some(height) = NonZeroUsize::new(cmp::max(level0, level1)) {
            self.stack[height] = ProductIter::new(
                if level0 > 0 {
                    // SAFETY: The level is greater than 0, so the node must be an inner node.
                    unsafe { node0.inner_children() }
                } else {
                    slice::from_ref(node0)
                },
                if level1 > 0 {
                    // SAFETY: The level is greater than 0, so the node must be an inner node.
                    unsafe { node1.inner_children() }
                } else {
                    slice::from_ref(node1)
                },
            );
        } else {
            // SAFETY: The level of both nodes is zero, so the nodes must be leaf nodes.
            *self.stack.leaf_mut() =
                unsafe { ProductIter::new(node0.leaf_children(), node1.leaf_children()) };
        }
    }

    /// # Safety
    ///
    /// The levels must be valid for the given nodes.
    pub(crate) unsafe fn new(
        filter: Filter,
        root0: &'a Node<N0, D0, Key0, Value0>,
        height0: usize,
        root1: &'b Node<N1, D1, Key1, Value1>,
        height1: usize,
    ) -> Self {
        if !filter.test_bounds(&root0.bounds, &root1.bounds) {
            return Self {
                stack: IterStack::empty_box(),
                padding0: 0,
                padding1: 0,
                filter,
                _phantom: PhantomData,
            };
        }

        let height = cmp::max(height0, height1);
        let mut iter = JoinIter {
            stack: IterStack::new_box(
                ProductIter::new(empty_slice(), empty_slice()),
                (0..height).map(|_| ProductIter::new(empty_slice(), empty_slice())),
            ),
            padding0: height - height0,
            padding1: height - height1,
            filter,
            _phantom: PhantomData,
        };
        // SAFETY: The levels are valid for the given nodes.
        unsafe {
            iter.start_join(root0, height0, root1, height1);
        }

        iter
    }
}

impl<
        'a,
        'b,
        N0,
        N1,
        const D0: usize,
        const D1: usize,
        Key0,
        Key1,
        Value0,
        Value1,
        Q0,
        Q1,
        Filter,
    > Iterator for JoinIter<'a, 'b, N0, N1, D0, D1, Key0, Key1, Value0, Value1, Q0, Q1, Filter>
where
    Key0: Borrow<Q0>,
    Key1: Borrow<Q1>,
    Q0: ?Sized,
    Q1: ?Sized,
    Filter: JoinFilter<N0, N1, D0, D1, Q0, Q1>,
{
    type Item = (&'a (Key0, Value0), &'b (Key1, Value1));

    fn next(&mut self) -> Option<Self::Item> {
        // The iterator operates on a stack of iterators joining the children of nodes, with the
        // iterator at the bottom of the stack joining the children of the root nodes, and the
        // iterator at the top of the stack joining the children of a pair of leaf nodes.
        //
        // The iterator at index 0 yields pairs of leaf node entries, while the remaining iterators
        // yield pairs of inner node entries. For the taller tree, the level of the nodes yielded by
        // the iterators is equal to the index of the iterator minus one. For the shorter tree, the
        // iterators are padded such that leaf nodes are yielded `padding` more times.
        let mut level = 0;
        while level < self.stack.len() {
            if let Some(nz_level) = NonZeroUsize::new(level) {
                if let Some((node0, node1)) = self.stack[nz_level]
                    .find(|(node0, node1)| self.filter.test_bounds(&node0.bounds, &node1.bounds))
                {
                    // SAFETY: For the taller tree, the level of the nodes yielded by the iterators
                    // is equal to the index of the iterator minus one. For the shorter tree, the
                    // iterators are padded such that leaf nodes are yielded `padding` more times.
                    level -= 1;
                    unsafe {
                        self.start_join(
                            node0,
                            level.saturating_sub(self.padding0),
                            node1,
                            level.saturating_sub(self.padding1),
                        );
                    }
                } else {
                    level += 1;
                }
            } else {
                if let Some((node0, node1)) = self.stack.leaf_mut().find(|(entry0, entry1)| {
                    self.filter.test_key(&entry0.0.borrow(), &entry1.0.borrow())
                }) {
                    return Some((node0, node1));
                } else {
                    level += 1;
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product_iter() {
        let a = [1, 2];
        let b = [4, 5];
        let mut iter = ProductIter::new(&a, &b);
        assert_eq!(iter.next(), Some((&a[0], &b[0])));
        assert_eq!(iter.next(), Some((&a[1], &b[0])));
        assert_eq!(iter.next(), Some((&a[0], &b[1])));
        assert_eq!(iter.next(), Some((&a[1], &b[1])));
        assert_eq!(iter.next(), None);
    }
}
