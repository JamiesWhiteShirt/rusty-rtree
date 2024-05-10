use core::slice;
use std::{borrow::Borrow, cmp, marker::PhantomData, num::NonZeroUsize, ptr};

use crate::{
    bounds::Bounded,
    filter::{JoinFilter, JoiningFilter},
    iter::{FilterIter, TreeIter},
    iter_stack::{InnerIterStack, IterStack, MaybeUninitIterStack},
    node::Node,
    rc_mut::{self, RcMutAlloc},
    util::empty_slice,
};

type MaybeUninitTreeIter<'a, B, Key, Value> =
    MaybeUninitIterStack<slice::Iter<'a, (Key, Value)>, slice::Iter<'a, Node<B, Key, Value>>>;

type RewindableTreeIter<'a, B, Key, Value> =
    IterStack<RewindableIter<'a, (Key, Value)>, RewindableIter<'a, Node<B, Key, Value>>>;

pub(crate) struct RewindableIter<'a, T> {
    slice: &'a [T],
    i: usize,
}

impl<'a, T> RewindableIter<'a, T> {
    fn new(slice: &'a [T]) -> Self {
        Self { slice, i: 0 }
    }

    fn rewind(&mut self) {
        self.i = 0;
    }

    fn remainder_iter(&self) -> slice::Iter<'a, T> {
        self.slice[self.i..].iter()
    }
}

impl<'a, T> Iterator for RewindableIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.slice.len() {
            let item = &self.slice[self.i];
            self.i += 1;
            Some(item)
        } else {
            None
        }
    }
}

pub struct LeftJoinIter<'l, 'r, BL, BR, KeyL, KeyR, ValueL, ValueR, QL, QR, Filter>
where
    QL: ?Sized,
    QR: ?Sized,
{
    left: Box<TreeIter<'l, BL, KeyL, ValueL>>,
    right: Box<RewindableTreeIter<'r, BR, KeyR, ValueR>>,
    right_alloc: RcMutAlloc<TreeIter<'r, BR, KeyR, ValueR>>,
    filter: Filter,
    _phantom: PhantomData<(&'l QL, &'r QR)>,
}

unsafe fn iterate_left_children<'a, B, Key, Value>(
    stack: &mut TreeIter<'a, B, Key, Value>,
    node: &'a Node<B, Key, Value>,
    level: usize,
) {
    if let Some(level) = NonZeroUsize::new(level) {
        stack[level] = unsafe { node.inner_children() }.iter();
    } else {
        *stack.leaf_mut() = unsafe { node.leaf_children() }.iter();
    }
}

unsafe fn iterate_right_children<'a, B, Key, Value>(
    stack: &mut RewindableTreeIter<'a, B, Key, Value>,
    node: Option<&'a Node<B, Key, Value>>,
    level: usize,
) {
    if let Some(level) = NonZeroUsize::new(level) {
        stack[level] = RewindableIter::new(
            node.map_or(empty_slice(), |node| unsafe { node.inner_children() }),
        );
    } else {
        *stack.leaf_mut() =
            RewindableIter::new(node.map_or(empty_slice(), |node| unsafe { node.leaf_children() }));
    }
}

fn rewind<'a, Leaf, T>(iter: &mut InnerIterStack<Leaf, RewindableIter<'a, T>>) {
    for i in 0..iter.len() {
        if i == iter.len() - 1 {
            iter[i].rewind();
        } else {
            iter[i] = RewindableIter::new(empty_slice());
        }
    }
}

impl<'l, 'r, BL, BR, ValueL, ValueR, KeyL, KeyR, QL, QR, Filter>
    LeftJoinIter<'l, 'r, BL, BR, KeyL, KeyR, ValueL, ValueR, QL, QR, Filter>
where
    QL: ?Sized,
    QR: ?Sized,
{
    pub(crate) unsafe fn new(
        filter: Filter,
        root_left: &'l Node<BL, KeyL, ValueL>,
        height_left: usize,
        root_right: &'r Node<BR, KeyR, ValueR>,
        height_right: usize,
    ) -> Self {
        let mut left = IterStack::new_box(
            empty_slice().iter(),
            (0..height_left + 1).map(|_| empty_slice().iter()),
        );
        left[NonZeroUsize::new(height_left + 1).unwrap()] = slice::from_ref(root_left).iter();

        let mut right = IterStack::new_box(
            RewindableIter::new(empty_slice()),
            (0..height_right + 1).map(|_| RewindableIter::new(empty_slice())),
        );
        right[NonZeroUsize::new(height_right + 1).unwrap()] =
            RewindableIter::new(slice::from_ref(root_right));

        let right_alloc =
            RcMutAlloc::new_for_value_raw(IterStack::from_raw_parts(ptr::null(), height_right + 2));

        Self {
            left,
            right,
            right_alloc,
            filter,
            _phantom: PhantomData,
        }
    }
}

impl<'l, 'r, BL, BR, KeyL, KeyR, ValueL, ValueR, QL, QR, Filter> Iterator
    for LeftJoinIter<'l, 'r, BL, BR, KeyL, KeyR, ValueL, ValueR, QL, QR, Filter>
where
    KeyL: Borrow<QL>,
    KeyR: Borrow<QR>,
    QL: ?Sized + Bounded<BL>,
    QR: ?Sized,
    Filter: JoinFilter<BL, BR, QL, QR> + Clone,
{
    type Item = (
        (&'l KeyL, &'l ValueL),
        FilterIter<
            rc_mut::RcMut<TreeIter<'r, BR, KeyR, ValueR>>,
            QR,
            JoiningFilter<'l, BL, QL, Filter>,
        >,
    );

    fn next(&mut self) -> Option<Self::Item> {
        let min_height = cmp::min(self.left.len(), self.right.len());
        if min_height == 0 {
            return None;
        }

        enum Op<'l, 'r, BL, BR, KeyL, KeyR, ValueL, ValueR> {
            Descend {
                left: &'l Node<BL, KeyL, ValueL>,
                right: Option<&'r Node<BR, KeyR, ValueR>>,
            },
            Ascend,
        }

        let mut level: usize = 0;
        loop {
            let op = if level > 0 {
                let iter_left = if level + 1 == self.right.len() {
                    &mut self.left[NonZeroUsize::new(level).unwrap()..]
                } else {
                    &mut self.left
                        [NonZeroUsize::new(level).unwrap()..NonZeroUsize::new(level + 1).unwrap()]
                };

                if let Some(node_left) =
                    iter_left.next(|node| unsafe { node.inner_children() }.iter())
                {
                    let iter_right = if level + 1 == self.left.len() {
                        &mut self.right[NonZeroUsize::new(level).unwrap()..]
                    } else {
                        &mut self.right[NonZeroUsize::new(level).unwrap()
                            ..NonZeroUsize::new(level + 1).unwrap()]
                    };

                    let node_right = iter_right.find(
                        |node| RewindableIter::new(unsafe { node.inner_children() }),
                        |node| self.filter.test_bounds(&node_left.bounds, &node.bounds),
                    );

                    Op::Descend {
                        left: node_left,
                        right: node_right,
                    }
                } else {
                    Op::Ascend
                }
            } else {
                let iter_left = self.left.leaf_mut();
                if let Some((key, value)) = iter_left.next() {
                    let stack = unsafe {
                        self.right_alloc.make_mut_init_raw(|stack_ptr| {
                            self.right.map_into(
                                |leaf| leaf.remainder_iter(),
                                |inner, _| inner.remainder_iter(),
                                (stack_ptr as *mut MaybeUninitTreeIter<'r, BR, KeyR, ValueR>)
                                    .as_mut()
                                    .unwrap_unchecked(),
                            );
                        })
                    };
                    let filter = JoiningFilter::new(key.borrow(), self.filter.clone());
                    return Some(((key, value), unsafe {
                        FilterIter::new_from_stack(stack, filter)
                    }));
                } else {
                    Op::Ascend
                }
            };

            match op {
                Op::Descend { left, right } => {
                    level -= 1;
                    unsafe { iterate_left_children(&mut self.left, left, level) };
                    unsafe { iterate_right_children(&mut self.right, right, level) };
                }
                Op::Ascend => {
                    level += 1;
                    if level == min_height {
                        break;
                    }
                    let iter_right = if level + 1 == self.left.len() {
                        &mut self.right[NonZeroUsize::new(level).unwrap()..]
                    } else {
                        &mut self.right[NonZeroUsize::new(level).unwrap()
                            ..NonZeroUsize::new(level + 1).unwrap()]
                    };
                    rewind(iter_right);
                }
            }
        }
        None
    }
}
