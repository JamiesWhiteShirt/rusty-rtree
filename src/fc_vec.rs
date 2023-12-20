use std::{
    alloc,
    fmt::{self, Debug},
    mem,
    ops::{Deref, DerefMut, Index, IndexMut},
    ptr::{self, NonNull},
    slice::{Iter, IterMut, SliceIndex},
};

/// A fixed-capacity vector whose capacity and operations are defined by a
/// [FCVecOps<T>]. Can be created via [FCVecOps<T>::new], and must be dropped
/// with [FCVecOps<T>::drop] before it is disposed of.
pub(crate) struct FCVec<T> {
    buf: NonNull<T>,
    len: usize,
}

impl<T> FCVec<T> {
    /// Returns the number of items in the vector.
    ///
    /// If the contents of the FCVec are dropped, the value returned is
    /// undefined.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn iter(&self) -> Iter<T> {
        self.deref().iter()
    }

    pub(crate) fn iter_mut(&mut self) -> IterMut<T> {
        self.deref_mut().iter_mut()
    }
}

impl<T> Deref for FCVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for FCVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.buf.as_ptr() as *mut T, self.len) }
    }
}

impl<T> PartialEq for FCVec<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }

    fn ne(&self, other: &Self) -> bool {
        self.len != other.len || self.iter().zip(other.iter()).any(|(a, b)| a != b)
    }
}

impl<T> Debug for FCVec<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T, I> Index<I> for FCVec<T>
where
    I: SliceIndex<[T]>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &(unsafe { std::slice::from_raw_parts(self.buf.as_ptr(), self.len) })[index]
    }
}

impl<T, I> IndexMut<I> for FCVec<T>
where
    I: SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut (unsafe { std::slice::from_raw_parts_mut(self.buf.as_ptr(), self.len) })[index]
    }
}

impl<'a, T> IntoIterator for &'a FCVec<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut FCVec<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Defines operations on [FCVec<T>] with a specified capacity.
///
/// # Safety
///
/// Operations are only safe on a given [FCVec<T>] if it was created by this
/// [FCVecOps] with [FCVecOps::new].
#[derive(Copy, Clone, Debug)]
pub(crate) struct FCVecOps {
    cap: usize,
}

impl FCVecOps {
    /// Creates a new [FCVecOps] with the specified capacity and type of items.
    pub(crate) fn new_ops(cap: usize) -> Self {
        FCVecOps { cap }
    }

    /// Allocates an empty [FCVec<T>] with the capacity specified by this
    /// FCVecOps in a safe [FCVecContainer<T>].
    pub(crate) fn new<T>(&self) -> FCVecContainer<T> {
        let layout = alloc::Layout::array::<T>(self.cap).unwrap();
        let buf = match NonNull::new(unsafe { alloc::alloc(layout) } as *mut T) {
            Some(ptr) => ptr,
            None => alloc::handle_alloc_error(layout),
        };
        FCVecContainer {
            data: FCVec { buf, len: 0 },
            ops: *self,
        }
    }

    /// Returns a [safe container](FCVecContainer<T>) for the given FCVec.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    pub(crate) unsafe fn wrap<T>(&self, vec: FCVec<T>) -> FCVecContainer<T> {
        FCVecContainer {
            data: vec,
            ops: *self,
        }
    }

    /// Returns a [safe reference](FCVecRef<T>) for the given FCVec.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    pub(crate) unsafe fn wrap_ref<'a, T>(&self, vec: &'a FCVec<T>) -> FCVecRef<'a, T> {
        FCVecRef {
            ops: *self,
            data: vec,
        }
    }

    /// Returns a [safe mutable reference](FCVecRefMut<T>) for the given FCVec.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    pub(crate) unsafe fn wrap_ref_mut<'a, T>(&self, vec: &'a mut FCVec<T>) -> FCVecRefMut<'a, T> {
        FCVecRefMut {
            ops: *self,
            data: vec,
        }
    }
}

pub(crate) struct FCVecRef<'a, T> {
    ops: FCVecOps,
    data: &'a FCVec<T>,
}

impl<'a, T> FCVecRef<'a, T> {
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns a copy of the referenced FCVec in a new FCVecContainer.
    pub(crate) fn clone(&self) -> FCVecContainer<T>
    where
        T: Clone,
    {
        let mut new = self.ops.new();
        for value in self.data {
            new.push(value.clone());
        }
        new
    }
}

impl<'a, T> Debug for FCVecRef<'a, T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FCVecRef")
            .field("cap", &self.ops.cap)
            .field("data", self.data)
            .finish()
    }
}

impl<'a, T> Deref for FCVecRef<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.data.deref()
    }
}

impl<'a, T, I> Index<I> for FCVecRef<'a, T>
where
    I: SliceIndex<[T]>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.data.index(index)
    }
}

impl<'a, T> IntoIterator for FCVecRef<'a, T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

pub(crate) struct FCVecRefMut<'a, T> {
    ops: FCVecOps,
    data: &'a mut FCVec<T>,
}

impl<'a, T> FCVecRefMut<'a, T> {
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn ops(&self) -> &FCVecOps {
        &self.ops
    }

    pub(crate) fn iter(&self) -> Iter<T> {
        self.data.iter()
    }

    pub(crate) fn iter_mut(&mut self) -> IterMut<T> {
        self.data.iter_mut()
    }

    /// Appends the given value to the end of the referenced FCVec.
    ///
    /// # Panics
    ///
    /// Panics if the FCVec is full.
    pub(crate) fn push(&mut self, value: T) {
        if self.data.len == self.ops.cap {
            panic!("Vector is full");
        }

        let buf = self.data.buf.as_ptr();
        unsafe { ptr::write(buf.add(self.data.len), value) };
        self.data.len += 1;
    }

    /// Appends the given value to the end of the referenced FCVec, returning
    /// the given value if the FCVec is full.
    pub(crate) fn try_push(&mut self, value: T) -> Option<T> {
        if self.data.len == self.ops.cap {
            Some(value)
        } else {
            let buf = self.data.buf.as_ptr();
            unsafe { ptr::write(buf.add(self.data.len), value) };
            self.data.len += 1;
            None
        }
    }

    /// Inserts the given value at the given index in the referenced FCVec.
    /// The value at the given index and all subsequent values are shifted
    /// right by one.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds or if the FCVec is full.
    pub(crate) fn insert(&mut self, index: usize, value: T) {
        if index > self.data.len {
            panic!("Index out of bounds");
        }
        if self.data.len == self.ops.cap {
            panic!("Vector is full");
        }

        let buf = self.data.buf.as_ptr();
        unsafe {
            ptr::copy(buf.add(index), buf.add(index + 1), self.data.len - index);
            ptr::write(buf.add(index), value);
        }
        self.data.len += 1;
    }

    /// Removes the value at the given index in the referenced FCVec and returns
    /// it. All subsequent values are shifted left by one.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub(crate) fn remove(&mut self, index: usize) -> T {
        if index >= self.data.len {
            panic!("Index out of bounds");
        }

        let buf = self.data.buf.as_ptr();
        let removed = unsafe { ptr::read(buf.add(index)) };
        unsafe {
            ptr::copy(
                buf.add(index + 1),
                buf.add(index),
                self.data.len - index - 1,
            )
        };
        self.data.len -= 1;
        removed
    }

    /// Removes the value at the given index in the referenced FCVec by swapping
    /// it with the last value in the FCVec and returning the removed value.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub(crate) fn swap_remove(&mut self, index: usize) -> T {
        if index >= self.data.len {
            panic!("Index out of bounds");
        }

        let buf = self.data.buf.as_ptr();
        let child = unsafe { ptr::read(buf.add(index)) };
        unsafe { ptr::copy(buf.add(self.data.len - 1), buf.add(index), 1) };
        self.data.len -= 1;
        child
    }

    /// Swaps the values at the given indices in the referenced FCVec.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    pub(crate) fn swap(&mut self, index_1: usize, index_2: usize) {
        if index_1 >= self.data.len || index_2 >= self.data.len {
            panic!("Index out of bounds");
        }

        let buf = self.data.buf.as_ptr();
        unsafe { ptr::swap(buf.add(index_1), buf.add(index_2)) };
    }

    /// Pops the last value from the referenced FCVec and returns it. Returns
    /// None if the FCVec is empty.
    fn pop(&mut self) -> Option<T> {
        if self.data.len == 0 {
            None
        } else {
            self.data.len -= 1;
            let buf = self.data.buf.as_ptr();
            Some(unsafe { ptr::read(buf.add(self.data.len)) })
        }
    }

    /// Drops the contents of the referenced [FCVec<T>]. After this, the FCVec
    /// is invalid and must not be used, but is safe to dispose of.
    pub(crate) unsafe fn drop(mut self) {
        // Drop all items in the FCVec
        while let Some(_) = self.pop() {}

        let layout = alloc::Layout::array::<T>(self.ops.cap).unwrap();
        alloc::dealloc(self.data.buf.as_ptr() as *mut u8, layout);
    }
}

impl<'a, T> Debug for FCVecRefMut<'a, T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FCVecRefMut")
            .field("cap", &self.ops.cap)
            .field("data", self.data)
            .finish()
    }
}

impl<'a, T> Deref for FCVecRefMut<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.data.deref()
    }
}

impl<'a, T> DerefMut for FCVecRefMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.deref_mut()
    }
}

impl<'a, T, I> Index<I> for FCVecRefMut<'a, T>
where
    I: SliceIndex<[T]>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.data.index(index)
    }
}

impl<'a, T, I> IndexMut<I> for FCVecRefMut<'a, T>
where
    I: SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.data.index_mut(index)
    }
}

impl<'a, T> IntoIterator for FCVecRefMut<'a, T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

pub(crate) struct FCVecContainer<T> {
    data: FCVec<T>,
    ops: FCVecOps,
}

impl<T> Debug for FCVecContainer<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FCVecContainer")
            .field("cap", &self.ops.cap)
            .field("data", &self.data)
            .finish()
    }
}

impl<T> Drop for FCVecContainer<T> {
    fn drop(&mut self) {
        unsafe {
            self.r#ref_mut().drop();
        }
    }
}

impl<T> Deref for FCVecContainer<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.data.deref()
    }
}

impl<T> DerefMut for FCVecContainer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.deref_mut()
    }
}

impl<T> IntoIterator for FCVecContainer<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let data = unsafe { ptr::read(&self.data) };
        let ops = unsafe { ptr::read(&self.ops) };
        mem::forget(self);

        let start = data.buf.as_ptr();
        let end = unsafe { start.add(data.len) };
        IntoIter {
            buf: data.buf,
            cap: ops.cap,
            start,
            end,
        }
    }
}

impl<'a, T> IntoIterator for &'a FCVecContainer<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut FCVecContainer<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<T> FCVecContainer<T> {
    /// Removes the FCVec from its container and returns it.
    ///
    /// # Safety
    ///
    /// The FCVec must be later wrapped using the container's [`FCVecOps`] in
    /// order to be used safely. It must also either be wrapped in an other
    /// container or be dropped with [`FCVecRef<T>::drop`].
    pub(crate) unsafe fn unwrap(self) -> FCVec<T> {
        let data = ptr::read(&self.data);
        mem::forget(self);
        data
    }

    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn r#ref(&self) -> FCVecRef<T> {
        unsafe { self.ops.wrap_ref(&self.data) }
    }

    pub(crate) fn r#ref_mut(&mut self) -> FCVecRefMut<T> {
        unsafe { self.ops.wrap_ref_mut(&mut self.data) }
    }

    pub(crate) fn push(&mut self, value: T) {
        self.r#ref_mut().push(value);
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
