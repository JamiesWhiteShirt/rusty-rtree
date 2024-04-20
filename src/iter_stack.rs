use std::{
    mem::ManuallyDrop,
    num::NonZeroUsize,
    ops::{Index, IndexMut},
    ptr,
};

use crate::util::empty_slice;

union Entry<Leaf, Inner> {
    leaf: ManuallyDrop<Leaf>,
    inner: ManuallyDrop<Inner>,
}

struct VecIterStack<Leaf, Inner>(Vec<Entry<Leaf, Inner>>);

impl<Leaf, Inner> VecIterStack<Leaf, Inner> {
    fn new<I>(leaf: Leaf, inner: I) -> Self
    where
        I: IntoIterator<Item = Inner>,
        I::IntoIter: ExactSizeIterator,
    {
        let inner = inner.into_iter();
        let mut stack = Vec::with_capacity(inner.len() + 1);
        stack.push(Entry {
            leaf: ManuallyDrop::new(leaf),
        });
        for inner in inner {
            stack.push(Entry {
                inner: ManuallyDrop::new(inner),
            });
        }
        Self(stack)
    }

    fn into_box(self) -> Box<IterStack<Leaf, Inner>> {
        let vec = unsafe { ptr::read(&ManuallyDrop::new(self).0) };
        let slice = Vec::leak(vec);
        // SAFETY: IterStack is repr(transparent) over [Entry<Leaf, Inner>]
        unsafe { Box::from_raw(slice as *mut _ as *mut IterStack<Leaf, Inner>) }
    }
}

impl<Leaf, Inner> Drop for VecIterStack<Leaf, Inner> {
    fn drop(&mut self) {
        if self.0.len() > 0 {
            unsafe { ManuallyDrop::drop(&mut self.0[0].leaf) }
        }
        for i in 1..self.0.len() {
            unsafe {
                ManuallyDrop::drop(&mut self.0[i].inner);
            }
        }
    }
}

#[repr(transparent)]
pub(crate) struct IterStack<Leaf, Inner>([Entry<Leaf, Inner>]);

impl<Leaf, Inner> IterStack<Leaf, Inner> {
    pub(crate) fn empty_box() -> Box<Self> {
        // SAFETY: IterStack is repr(transparent) over [Entry<Leaf, Inner>]
        unsafe { Box::from_raw(empty_slice::<Entry<Leaf, Inner>>() as *mut _ as *mut Self) }
    }

    pub(crate) fn new_box<I>(leaf: Leaf, inner: I) -> Box<Self>
    where
        I: IntoIterator<Item = Inner>,
        I::IntoIter: ExactSizeIterator,
    {
        VecIterStack::new(leaf, inner).into_box()
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn leaf(&self) -> &Leaf {
        unsafe { &self.0[0].leaf }
    }

    pub(crate) fn leaf_mut(&mut self) -> &mut Leaf {
        unsafe { &mut self.0[0].leaf }
    }
}

impl<Leaf, Inner> Index<NonZeroUsize> for IterStack<Leaf, Inner> {
    type Output = Inner;

    fn index(&self, index: NonZeroUsize) -> &Self::Output {
        unsafe { &self.0[index.get()].inner }
    }
}

impl<Leaf, Inner> IndexMut<NonZeroUsize> for IterStack<Leaf, Inner> {
    fn index_mut(&mut self, index: NonZeroUsize) -> &mut Self::Output {
        unsafe { &mut self.0[index.get()].inner }
    }
}

impl<Leaf, Inner> Drop for IterStack<Leaf, Inner> {
    fn drop(&mut self) {
        if self.0.len() > 0 {
            unsafe { ManuallyDrop::drop(&mut self.0[0].leaf) }
        }
        for i in 1..self.0.len() {
            unsafe {
                ManuallyDrop::drop(&mut self.0[i].inner);
            }
        }
    }
}
