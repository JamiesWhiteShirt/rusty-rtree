use std::{
    alloc,
    fmt::{self, Debug},
    mem::{self, ManuallyDrop},
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

    /// Removes the value at the given index in the given FCVec by swapping it
    /// with the last value in the FCVec and returning the removed value.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub(crate) fn swap_remove(&mut self, index: usize) -> T {
        if index >= self.len {
            panic!("Index out of bounds");
        }

        let buf = self.buf.as_ptr();
        let child = unsafe { ptr::read(buf.add(index)) };
        unsafe {
            ptr::copy(buf.add(self.len - 1), buf.add(index), 1);
        }
        self.len -= 1;
        child
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
pub(crate) struct FCVecOps {
    cap: usize,
}

impl FCVecOps {
    /// Creates a new [FCVecOps] with the specified capacity and type of items.
    pub(crate) fn new_ops(cap: usize) -> Self {
        FCVecOps { cap }
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
    pub(crate) fn new<'a, T>(&'a self) -> FCVecContainer<'a, T> {
        let layout = alloc::Layout::array::<T>(self.cap).unwrap();
        let buf = match NonNull::new(unsafe { alloc::alloc(layout) } as *mut T) {
            Some(ptr) => ptr,
            None => alloc::handle_alloc_error(layout),
        };
        FCVecContainer {
            data: FCVec { buf, len: 0 },
            ops: self,
        }
    }

    /// Drops the contents of the given [FCVec<T>]. After this, the FCVec is
    /// invalid and must not be used, but is safe to dispose of.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps, and must not
    /// be already dropped.
    unsafe fn drop<T>(&self, data: &mut FCVec<T>) {
        let layout = alloc::Layout::array::<T>(self.cap).unwrap();
        alloc::dealloc(data.buf.as_ptr() as *mut u8, layout);
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
    unsafe fn push<T>(&self, data: &mut FCVec<T>, value: T) {
        if data.len == self.cap {
            panic!("Vector is full");
        }

        let buf = data.buf.as_ptr();
        ptr::write(buf.add(data.len), value);
        data.len += 1;
    }

    /// Appends the given value to the end of the given FCVec, returning the
    /// given value if the FCVec is full.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    unsafe fn try_push<T>(&self, data: &mut FCVec<T>, value: T) -> Option<T> {
        if data.len == self.cap {
            Some(value)
        } else {
            let buf = data.buf.as_ptr();
            ptr::write(buf.add(data.len), value);
            data.len += 1;
            None
        }
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
    unsafe fn insert<T>(&self, data: &mut FCVec<T>, index: usize, value: T) {
        if index > data.len {
            panic!("Index out of bounds");
        }
        if data.len == self.cap {
            panic!("Vector is full");
        }

        let buf = data.buf.as_ptr();
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
    unsafe fn remove<T>(&self, data: &mut FCVec<T>, index: usize) -> T {
        if index >= data.len {
            panic!("Index out of bounds");
        }

        let buf = data.buf.as_ptr();
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
    unsafe fn swap_remove<T>(&self, data: &mut FCVec<T>, index: usize) -> T {
        if index >= data.len {
            panic!("Index out of bounds");
        }

        let buf = data.buf.as_ptr();
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
    unsafe fn swap<T>(&self, data: &mut FCVec<T>, index_1: usize, index_2: usize) {
        if index_1 >= data.len || index_2 >= data.len {
            panic!("Index out of bounds");
        }

        let buf = data.buf.as_ptr();
        ptr::swap(buf.add(index_1), buf.add(index_2));
    }

    /// Returns a [safe container](FCVecContainer<T>) for the given FCVec.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    pub(crate) unsafe fn wrap<'a, T>(&'a self, vec: FCVec<T>) -> FCVecContainer<'a, T> {
        FCVecContainer {
            data: vec,
            ops: self,
        }
    }

    /// Returns a [safe reference](FCVecRef<T>) for the given FCVec.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    pub(crate) unsafe fn wrap_ref<'a, 'b, T>(&'a self, vec: &'b FCVec<T>) -> FCVecRef<'a, 'b, T> {
        FCVecRef {
            ops: self,
            data: vec,
        }
    }

    /// Returns a [safe mutable reference](FCVecRefMut<T>) for the given FCVec.
    ///
    /// # Safety
    ///
    /// The given FCVec must have been created by this FCVecOps and must not
    /// have been dropped.
    pub(crate) unsafe fn wrap_ref_mut<'a, 'b, T>(
        &'a self,
        vec: &'b mut FCVec<T>,
    ) -> FCVecRefMut<'a, 'b, T> {
        FCVecRefMut {
            ops: self,
            data: vec,
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
    unsafe fn clone<T>(&self, vec: &FCVec<T>) -> FCVecContainer<T>
    where
        T: Clone,
    {
        let mut new = self.new();
        for value in vec.iter() {
            new.push(value.clone());
        }
        new
    }

    unsafe fn into_iter<T>(&self, vec: FCVec<T>) -> IntoIter<T> {
        let start = vec.buf.as_ptr();
        let end = start.add(vec.len);
        IntoIter {
            buf: vec.buf,
            cap: self.cap,
            start,
            end,
        }
    }
}

pub(crate) struct FCVecRef<'a, 'b, T> {
    ops: &'a FCVecOps,
    data: &'b FCVec<T>,
}

impl<'a, 'b, T> FCVecRef<'a, 'b, T> {
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn iter(&self) -> Iter<T> {
        self.data.iter()
    }

    pub(crate) fn clone(&self) -> FCVecContainer<'a, T>
    where
        T: Clone,
    {
        unsafe { self.ops.clone(self.data) }
    }
}

impl<'a, 'b, T> Debug for FCVecRef<'a, 'b, T>
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

impl<'a, 'b, T> Deref for FCVecRef<'a, 'b, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.data.deref()
    }
}

impl<'a, 'b, T, I> Index<I> for FCVecRef<'a, 'b, T>
where
    I: SliceIndex<[T]>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.data.index(index)
    }
}

impl<'a, 'b, T> IntoIterator for FCVecRef<'a, 'b, T> {
    type Item = &'b T;
    type IntoIter = Iter<'b, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

pub(crate) struct FCVecRefMut<'a, 'b, T> {
    ops: &'a FCVecOps,
    data: &'b mut FCVec<T>,
}

impl<'a, 'b, T> FCVecRefMut<'a, 'b, T> {
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn ops(&self) -> &'a FCVecOps {
        self.ops
    }

    pub(crate) fn iter(&self) -> Iter<T> {
        self.data.iter()
    }

    pub(crate) fn iter_mut(&mut self) -> IterMut<T> {
        self.data.iter_mut()
    }

    pub(crate) fn iter_mut_take(self) -> IterMut<'b, T> {
        self.data.iter_mut()
    }

    pub(crate) fn swap_remove(&mut self, index: usize) -> T {
        unsafe { self.ops.swap_remove(self.data, index) }
    }

    pub(crate) fn push(&mut self, value: T) {
        unsafe { self.ops.push(self.data, value) }
    }

    pub(crate) fn try_push(&mut self, value: T) -> Option<T> {
        unsafe { self.ops.try_push(self.data, value) }
    }

    pub(crate) fn insert(&mut self, index: usize, value: T) {
        unsafe { self.ops.insert(self.data, index, value) }
    }

    pub(crate) fn remove(&mut self, index: usize) -> T {
        unsafe { self.ops.remove(self.data, index) }
    }

    pub(crate) fn swap(&mut self, index_1: usize, index_2: usize) {
        unsafe { self.ops.swap(self.data, index_1, index_2) }
    }

    pub(crate) fn clone(&self) -> FCVecContainer<'a, T>
    where
        T: Clone,
    {
        unsafe { self.ops.clone(self.data) }
    }
}

impl<'a, 'b, T> Debug for FCVecRefMut<'a, 'b, T>
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

impl<'a, 'b, T> Deref for FCVecRefMut<'a, 'b, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.data.deref()
    }
}

impl<'a, 'b, T> DerefMut for FCVecRefMut<'a, 'b, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.deref_mut()
    }
}

impl<'a, 'b, T, I> Index<I> for FCVecRefMut<'a, 'b, T>
where
    I: SliceIndex<[T]>,
{
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.data.index(index)
    }
}

impl<'a, 'b, T, I> IndexMut<I> for FCVecRefMut<'a, 'b, T>
where
    I: SliceIndex<[T]>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.data.index_mut(index)
    }
}

impl<'a, 'b, T> IntoIterator for FCVecRefMut<'a, 'b, T> {
    type Item = &'b mut T;
    type IntoIter = IterMut<'b, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

pub(crate) struct FCVecContainer<'a, T> {
    data: FCVec<T>,
    ops: &'a FCVecOps,
}

impl<'a, T> Debug for FCVecContainer<'a, T>
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
        self.data.deref()
    }
}

impl<'a, T> DerefMut for FCVecContainer<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.deref_mut()
    }
}

impl<'a, T> IntoIterator for FCVecContainer<'a, T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let from = ManuallyDrop::new(self);
        let from = &*from as *const FCVecContainer<T>;

        unsafe {
            let ops = ptr::read(&(*from).ops);
            let data = ptr::read(&(*from).data);

            ops.into_iter(data)
        }
    }
}

impl<'a, T> IntoIterator for &'a FCVecContainer<'a, T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T> FCVecContainer<'a, T> {
    /// Removes the FCVec from the container and returns it.
    ///
    /// # Safety
    ///
    /// The FCVec must be later wrapped with [FCVecOps<T>::wrap] in order to
    /// be used safely. It must also be dropped with [FCVecOps<T>::drop] before
    /// it is disposed of.
    pub(crate) unsafe fn unwrap(self) -> FCVec<T> {
        let data = ptr::read(&self.data);
        mem::forget(self);
        data
    }

    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn iter(&self) -> Iter<T> {
        self.data.iter()
    }

    pub(crate) fn iter_mut(&mut self) -> IterMut<T> {
        self.data.iter_mut()
    }

    pub(crate) fn swap_remove(&mut self, index: usize) -> T {
        unsafe { self.ops.swap_remove(&mut self.data, index) }
    }

    pub(crate) fn push(&mut self, value: T) {
        unsafe { self.ops.push(&mut self.data, value) }
    }

    pub(crate) fn try_push(&mut self, value: T) -> Option<T> {
        unsafe { self.ops.try_push(&mut self.data, value) }
    }

    pub(crate) fn insert(&mut self, index: usize, value: T) {
        unsafe { self.ops.insert(&mut self.data, index, value) }
    }

    pub(crate) fn remove(&mut self, index: usize) -> T {
        unsafe { self.ops.remove(&mut self.data, index) }
    }

    pub(crate) fn swap(&mut self, index_1: usize, index_2: usize) {
        unsafe { self.ops.swap(&mut self.data, index_1, index_2) }
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
