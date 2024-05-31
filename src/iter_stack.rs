use std::{
    mem::{ManuallyDrop, MaybeUninit},
    num::NonZeroUsize,
    ops::{Index, IndexMut, Range, RangeFrom, RangeTo},
    ptr,
};

use crate::util::empty_slice;

pub(crate) union Entry<Leaf, Inner> {
    pub(crate) leaf: ManuallyDrop<Leaf>,
    pub(crate) inner: ManuallyDrop<Inner>,
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
        if !self.0.is_empty() {
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
pub struct IterStack<Leaf, Inner>([Entry<Leaf, Inner>]);

#[repr(transparent)]
pub(crate) struct MaybeUninitIterStack<Leaf, Inner>([MaybeUninit<Entry<Leaf, Inner>>]);

impl<Leaf, Inner> IterStack<Leaf, Inner> {
    pub(crate) const fn from_raw_parts(data: *const Entry<Leaf, Inner>, len: usize) -> *const Self {
        ptr::slice_from_raw_parts(data, len) as *mut Self
    }

    pub(crate) unsafe fn init_into(
        entries: &mut MaybeUninitIterStack<Leaf, Inner>,
        leaf: impl FnOnce() -> Leaf,
        mut inner: impl FnMut(NonZeroUsize) -> Inner,
    ) -> &mut Self {
        struct Guard<'a, Leaf, Inner> {
            target: &'a mut MaybeUninitIterStack<Leaf, Inner>,
            i: usize,
        }

        impl<'a, Leaf, Inner> Guard<'a, Leaf, Inner> {
            fn len(&self) -> usize {
                self.target.0.len()
            }

            fn push(&mut self, entry: Entry<Leaf, Inner>) {
                assert!(
                    self.i < self.target.0.len(),
                    "Too many entries have been initialized"
                );
                self.target.0[self.i].write(entry);
                self.i += 1;
            }

            fn finish(mut self) -> &'a mut IterStack<Leaf, Inner> {
                assert_eq!(
                    self.i,
                    self.target.0.len(),
                    "Not all entries have been initialized"
                );

                // All entries have been initialized, so don't drop them
                self.i = 0;
                unsafe { &mut *(self.target as *mut _ as *mut IterStack<Leaf, Inner>) }
            }
        }

        impl<Leaf, Inner> Drop for Guard<'_, Leaf, Inner> {
            fn drop(&mut self) {
                for i in 0..self.i {
                    unsafe {
                        self.target.0[i].assume_init_drop();
                    }
                }
            }
        }

        let mut _guard = Guard {
            target: entries,
            i: 0,
        };

        if _guard.len() > 0 {
            _guard.push(Entry {
                leaf: ManuallyDrop::new(leaf()),
            });
            for i in 1.._guard.len() {
                _guard.push(Entry {
                    inner: ManuallyDrop::new(inner(NonZeroUsize::new(i).unwrap())),
                });
            }
        }

        _guard.finish()
    }

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

    #[allow(dead_code)]
    pub(crate) fn map<Inner1, Leaf1>(
        &self,
        map_leaf: impl FnOnce(&Leaf) -> Leaf1,
        mut map_inner: impl FnMut(&Inner, NonZeroUsize) -> Inner1,
    ) -> Box<IterStack<Leaf1, Inner1>> {
        let mut stack = Vec::with_capacity(self.0.len());
        if let Some(len) = NonZeroUsize::new(self.0.len()) {
            stack.push(Entry {
                leaf: ManuallyDrop::new(map_leaf(unsafe { &self.0[0].leaf })),
            });
            for i in 1..len.get() {
                stack.push(Entry {
                    inner: ManuallyDrop::new(map_inner(
                        unsafe { &self.0[i].inner },
                        NonZeroUsize::new(i).unwrap(),
                    )),
                });
            }
        }
        unsafe { Box::from_raw(Vec::leak(stack) as *mut _ as *mut IterStack<Leaf1, Inner1>) }
    }

    pub(crate) unsafe fn map_into<'a, Leaf1, Inner1>(
        &self,
        map_leaf: impl FnOnce(&Leaf) -> Leaf1,
        mut map_inner: impl FnMut(&Inner, NonZeroUsize) -> Inner1,
        target: &'a mut MaybeUninitIterStack<Leaf1, Inner1>,
    ) -> &'a mut IterStack<Leaf1, Inner1> {
        assert_eq!(
            target.0.len(),
            self.0.len(),
            "Cannot map into a different length"
        );

        unsafe {
            IterStack::<Leaf1, Inner1>::init_into(
                target,
                || map_leaf(self.leaf()),
                |i| map_inner(&self[i], i),
            )
        }
    }
}

impl<Leaf, Inner> Index<RangeTo<usize>> for IterStack<Leaf, Inner> {
    type Output = IterStack<Leaf, Inner>;

    fn index(&self, index: RangeTo<usize>) -> &Self::Output {
        unsafe { &*((&self.0[..index.end]) as *const _ as *const IterStack<Leaf, Inner>) }
    }
}

impl<Leaf, Inner> IndexMut<RangeTo<usize>> for IterStack<Leaf, Inner> {
    fn index_mut(&mut self, index: RangeTo<usize>) -> &mut Self::Output {
        unsafe { &mut *((&mut self.0[..index.end]) as *mut _ as *mut IterStack<Leaf, Inner>) }
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

impl<Leaf, Inner> Index<Range<NonZeroUsize>> for IterStack<Leaf, Inner> {
    type Output = InnerIterStack<Leaf, Inner>;

    fn index(&self, index: Range<NonZeroUsize>) -> &Self::Output {
        unsafe {
            &*((&self.0[index.start.get()..index.end.get()]) as *const _
                as *const InnerIterStack<Leaf, Inner>)
        }
    }
}

impl<Leaf, Inner> IndexMut<Range<NonZeroUsize>> for IterStack<Leaf, Inner> {
    fn index_mut(&mut self, index: Range<NonZeroUsize>) -> &mut Self::Output {
        unsafe {
            &mut *((&mut self.0[index.start.get()..index.end.get()]) as *mut _
                as *mut InnerIterStack<Leaf, Inner>)
        }
    }
}

impl<Leaf, Inner> Index<RangeFrom<NonZeroUsize>> for IterStack<Leaf, Inner> {
    type Output = InnerIterStack<Leaf, Inner>;

    fn index(&self, index: RangeFrom<NonZeroUsize>) -> &Self::Output {
        unsafe {
            &*((&self.0[index.start.get()..]) as *const _ as *const InnerIterStack<Leaf, Inner>)
        }
    }
}

impl<Leaf, Inner> IndexMut<RangeFrom<NonZeroUsize>> for IterStack<Leaf, Inner> {
    fn index_mut(&mut self, index: RangeFrom<NonZeroUsize>) -> &mut Self::Output {
        unsafe {
            &mut *((&mut self.0[index.start.get()..]) as *mut _ as *mut InnerIterStack<Leaf, Inner>)
        }
    }
}

impl<Leaf, Inner> Index<RangeTo<NonZeroUsize>> for IterStack<Leaf, Inner> {
    type Output = IterStack<Leaf, Inner>;

    fn index(&self, index: RangeTo<NonZeroUsize>) -> &Self::Output {
        unsafe { &*((&self.0[..index.end.get()]) as *const _ as *const IterStack<Leaf, Inner>) }
    }
}

impl<Leaf, Inner> IndexMut<RangeTo<NonZeroUsize>> for IterStack<Leaf, Inner> {
    fn index_mut(&mut self, index: RangeTo<NonZeroUsize>) -> &mut Self::Output {
        unsafe { &mut *((&mut self.0[..index.end.get()]) as *mut _ as *mut IterStack<Leaf, Inner>) }
    }
}

impl<Leaf, Inner> Drop for IterStack<Leaf, Inner> {
    fn drop(&mut self) {
        if !self.0.is_empty() {
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
pub struct InnerIterStack<Leaf, Inner>([Entry<Leaf, Inner>]);

impl<Leaf, Inner> InnerIterStack<Leaf, Inner> {
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
}

impl<Leaf, Inner> InnerIterStack<Leaf, Inner>
where
    Inner: Iterator,
{
    pub(crate) fn next(
        &mut self,
        mut get_children: impl FnMut(Inner::Item) -> Inner,
    ) -> Option<Inner::Item> {
        let mut i = 0;
        while i < self.0.len() {
            let iter = &mut self[i];
            if let Some(child) = iter.next() {
                if i == 0 {
                    return Some(child);
                }
                i -= 1;
                self[i] = get_children(child);
            } else {
                i += 1;
            }
        }
        None
    }

    pub(crate) fn find(
        &mut self,
        mut get_children: impl FnMut(Inner::Item) -> Inner,
        mut predicate: impl FnMut(&Inner::Item) -> bool,
    ) -> Option<Inner::Item> {
        let mut i = 0;
        while i < self.0.len() {
            let iter = &mut self[i];
            if let Some(child) = iter.find(&mut predicate) {
                if i == 0 {
                    return Some(child);
                }
                i -= 1;
                self[i] = get_children(child);
            } else {
                i += 1;
            }
        }
        None
    }
}

impl<Leaf, Inner> Index<usize> for InnerIterStack<Leaf, Inner> {
    type Output = Inner;

    fn index(&self, index: usize) -> &Self::Output {
        unsafe { &self.0[index].inner }
    }
}

impl<Leaf, Inner> IndexMut<usize> for InnerIterStack<Leaf, Inner> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { &mut self.0[index].inner }
    }
}

impl<Leaf, Inner> Drop for InnerIterStack<Leaf, Inner> {
    fn drop(&mut self) {
        for i in 0..self.0.len() {
            unsafe {
                ManuallyDrop::drop(&mut self.0[i].inner);
            }
        }
    }
}
