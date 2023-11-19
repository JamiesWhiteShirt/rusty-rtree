use std::{
    alloc,
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
};

/// A fixed-capacity vector whose type of items, capacity and operations are
/// defined by a [FCVecOps<T>]. Can be created via [FCVecOps<T>::new], and must
/// be dropped with [FCVecOps<T>::drop] before it is disposed of.
pub(crate) struct FCVec {
    buf: NonNull<u8>,
    len: usize,
}

impl FCVec {
    /// Returns the number of items in the vector.
    ///
    /// If the contents of the FCVec are dropped, the value returned is
    /// undefined.
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

/// Defines operations on [FCVec<T>] with a specified capacity and type of
/// items.
///
/// # Safety
///
/// Operations are only safe on a given [FCVec<T>] if it was created by this
/// [FCVecOps<T>] with [FCVecOps<T>::new].
pub(crate) struct FCVecOps<T> {
    cap: usize,

    _phantom: PhantomData<T>,
}

impl<T> Copy for FCVecOps<T> {}
impl<T> Clone for FCVecOps<T> {
    fn clone(&self) -> Self {
        FCVecOps {
            cap: self.cap,
            _phantom: PhantomData,
        }
    }
}

impl<T> FCVecOps<T> {
    /// Creates a new [FCVecOps<T>] with the specified capacity and type of items.
    pub(crate) fn new_ops(cap: usize) -> Self {
        FCVecOps {
            cap,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn cap(&self) -> usize {
        self.cap
    }

    /// Allocates an empty [FCVec<T>] with the capacity and type specified by
    /// this FCVecOps. The returned FCVec is safe to use with the FCVecOps that
    /// created it.
    ///
    /// # Safety
    ///
    /// The returned FCVec can only be used with the same FCVecOps that created
    /// it, and its contents must be dropped with [FCVecOps<T>::drop] before the
    /// FCVec is disposed of.
    pub(crate) unsafe fn new(&self) -> FCVec {
        let layout = alloc::Layout::array::<T>(self.cap).unwrap();
        let buf = match NonNull::new(alloc::alloc(layout)) {
            Some(ptr) => ptr,
            None => alloc::handle_alloc_error(layout),
        };
        FCVec { buf, len: 0 }
    }

    /// Drops the contents of the given [FCVec<T>]. After this, the FCVec is
    /// invalid and must not be used, but is safe to dispose of.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps, and must not
    /// be already dropped.
    pub(crate) unsafe fn drop(&self, data: &mut FCVec) {
        let layout = alloc::Layout::array::<T>(self.cap).unwrap();
        alloc::dealloc(data.buf.as_ptr(), layout);
    }

    /// Returns a slice of the given FCVec.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    pub(crate) unsafe fn as_slice<'a>(&self, data: &'a FCVec) -> &'a [T] {
        std::slice::from_raw_parts(data.buf.as_ptr() as *const T, data.len)
    }

    /// Returns a mutable slice of the given FCVec.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    pub(crate) unsafe fn as_slice_mut<'a>(&self, data: &'a mut FCVec) -> &'a mut [T] {
        std::slice::from_raw_parts_mut(data.buf.as_ptr() as *mut T, data.len)
    }

    /// Appends the given value to the end of the given FCVec.
    ///
    /// # Panics
    ///
    /// Panics if the FCVec is full.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    pub(crate) unsafe fn push(&self, data: &mut FCVec, value: T) {
        if data.len == self.cap {
            panic!("Vector is full");
        }

        let buf = data.buf.as_ptr() as *mut T;
        ptr::write(buf.add(data.len), value);
        data.len += 1;
    }

    /// Inserts the given value at the given index in the given FCVec.
    /// The value at the given index and all subsequent values are shifted
    /// right by one.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds or if the FCVec is full.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    pub(crate) unsafe fn insert(&self, data: &mut FCVec, index: usize, value: T) {
        if index > data.len {
            panic!("Index out of bounds");
        }
        if data.len == self.cap {
            panic!("Vector is full");
        }

        let buf = data.buf.as_ptr() as *mut T;
        ptr::copy(buf.add(index), buf.add(index + 1), data.len - index);
        ptr::write(buf.add(index), value);
        data.len += 1;
    }

    /// Removes the value at the given index in the given FCVec and returns it.
    /// All subsequent values are shifted left by one.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    pub(crate) unsafe fn remove(&self, data: &mut FCVec, index: usize) -> T {
        if index >= data.len {
            panic!("Index out of bounds");
        }

        let buf = data.buf.as_ptr() as *mut T;
        let removed = ptr::read(buf.add(index));
        ptr::copy(buf.add(index + 1), buf.add(index), data.len - index - 1);
        data.len -= 1;
        removed
    }

    /// Removes the value at the given index in the given FCVec by swapping it
    /// with the last value in the FCVec and returning the removed value.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    pub(crate) unsafe fn swap_remove(&self, data: &mut FCVec, index: usize) -> T {
        if index >= data.len {
            panic!("Index out of bounds");
        }

        let buf = data.buf.as_ptr() as *mut T;
        let child = ptr::read(buf.add(index));
        ptr::copy(buf.add(data.len - 1), buf.add(index), 1);
        data.len -= 1;
        child
    }

    /// Swaps the values at the given indices in the given FCVec.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    pub(crate) unsafe fn swap(&self, data: &mut FCVec, index_1: usize, index_2: usize) {
        if index_1 >= data.len || index_2 >= data.len {
            panic!("Index out of bounds");
        }

        let buf = data.buf.as_ptr() as *mut T;
        ptr::swap(buf.add(index_1), buf.add(index_2));
    }

    /// Returns a reference to the value at the given index in the given FCVec.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    pub(crate) unsafe fn at<'a>(&self, data: &'a FCVec, index: usize) -> &'a T {
        if index >= data.len {
            panic!("Index out of bounds");
        }

        &*(data.buf.as_ptr() as *const T).add(index)
    }

    /// Returns a mutable reference to the value at the given index in the given
    /// FCVec.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    pub(crate) unsafe fn at_mut<'a>(&self, data: &'a mut FCVec, index: usize) -> &'a mut T {
        if index >= data.len {
            panic!("Index out of bounds");
        }

        &mut *(data.buf.as_ptr() as *mut T).add(index)
    }

    /// Returns a [safe container](FCVecContainer<T>) for the given FCVec.
    pub(crate) unsafe fn wrap<'a>(&'a self, vec: FCVec) -> FCVecContainer<'a, T> {
        FCVecContainer {
            data: vec,
            ops: self,
        }
    }

    /// Clones the given FCVec. The returned FCVec is safe to use with the
    /// FCVecOps that created it.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    ///
    /// The returned FCVec can only be used with the same FCVecOps that created
    /// it, and its contents must be dropped with [FCVecOps<T>::drop] before the
    /// FCVec is disposed of.
    pub(crate) unsafe fn clone(self, vec: &FCVec) -> FCVec
    where
        T: Clone,
    {
        let mut new = self.new();
        for value in self.as_slice(&vec) {
            self.push(&mut new, value.clone());
        }
        new
    }
}

pub(crate) struct Iter<'a, T: 'a> {
    data: &'a FCVec,
    index: usize,

    _phantom: PhantomData<T>,
}

impl<'a, T: 'a> Iter<'a, T> {
    pub(crate) unsafe fn new(data: &'a FCVec) -> Iter<'a, T> {
        Iter {
            data,
            index: 0,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: 'a> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.data.len {
            unsafe {
                let result = &*(self.data.buf.as_ptr() as *const T).add(self.index);
                self.index += 1;
                Some(result)
            }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.data.len - self.index;
        (len, Some(len))
    }
}

pub(crate) struct FCVecContainer<'a, T> {
    data: FCVec,
    ops: &'a FCVecOps<T>,
}

impl<'a, T> Drop for FCVecContainer<'a, T> {
    fn drop(&mut self) {
        unsafe {
            self.ops.drop(&mut self.data);
        }
    }
}

impl<'a, T> Deref for FCVecContainer<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { self.ops.as_slice(&self.data) }
    }
}

impl<'a, T> DerefMut for FCVecContainer<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ops.as_slice_mut(&mut self.data) }
    }
}

impl<'a, T> IntoIterator for FCVecContainer<'a, T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        unsafe {
            let from = ManuallyDrop::new(self);

            let start = from.data.buf.as_ptr() as *const T;
            let end = start.add(from.data.len);
            IntoIter {
                buf: from.data.buf.cast::<T>(),
                cap: from.ops.cap,
                start,
                end,
            }
        }
    }
}

pub(crate) struct IntoIter<T> {
    buf: NonNull<T>,
    cap: usize,
    start: *const T,
    end: *const T,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            None
        } else {
            unsafe {
                let result = ptr::read(self.start);
                self.start = self.start.add(1);
                Some(result)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (self.end as usize - self.start as usize) / std::mem::size_of::<T>();
        (len, Some(len))
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start == self.end {
            None
        } else {
            unsafe {
                self.end = self.end.sub(1);
                let result = ptr::read(self.end);
                Some(result)
            }
        }
    }
}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        for _ in &mut *self {}
        let layout = alloc::Layout::array::<T>(self.cap).unwrap();
        unsafe {
            alloc::dealloc(self.buf.as_ptr() as *mut u8, layout);
        }
    }
}
