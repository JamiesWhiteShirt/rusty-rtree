use std::{
    alloc,
    marker::PhantomData,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
};

pub(crate) struct FSVecData {
    buf: NonNull<u8>,
    len: usize,
}

impl FSVecData {
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

#[derive(Clone, Copy)]
pub(crate) struct FSVecOps<T> {
    cap: usize,

    _phantom: PhantomData<T>,
}

impl<T> FSVecOps<T> {
    pub(crate) fn new_ops(cap: usize) -> Self {
        FSVecOps {
            cap,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn cap(&self) -> usize {
        self.cap
    }

    pub(crate) unsafe fn new(&self) -> FSVecData {
        let layout = alloc::Layout::array::<T>(self.cap).unwrap();
        let buf = match NonNull::new(alloc::alloc(layout)) {
            Some(ptr) => ptr,
            None => alloc::handle_alloc_error(layout),
        };
        FSVecData { buf, len: 0 }
    }

    pub(crate) unsafe fn drop(&self, data: &mut FSVecData) {
        let layout = alloc::Layout::array::<T>(self.cap).unwrap();
        alloc::dealloc(data.buf.as_ptr(), layout);
    }

    pub(crate) unsafe fn as_slice<'a>(&self, data: &'a FSVecData) -> &'a [T] {
        std::slice::from_raw_parts(data.buf.as_ptr() as *const T, data.len)
    }

    pub(crate) unsafe fn as_slice_mut<'a>(&self, data: &'a mut FSVecData) -> &'a mut [T] {
        std::slice::from_raw_parts_mut(data.buf.as_ptr() as *mut T, data.len)
    }

    pub(crate) unsafe fn push(&self, data: &mut FSVecData, value: T) {
        if data.len == self.cap {
            panic!("Vector is full");
        }

        let buf = data.buf.as_ptr() as *mut T;
        ptr::write(buf.add(data.len), value);
        data.len += 1;
    }

    pub(crate) unsafe fn insert(&self, data: &mut FSVecData, index: usize, value: T) {
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

    pub(crate) unsafe fn remove(&self, data: &mut FSVecData, index: usize) -> T {
        if index >= data.len {
            panic!("Index out of bounds");
        }

        let buf = data.buf.as_ptr() as *mut T;
        let removed = ptr::read(buf.add(index));
        ptr::copy(buf.add(index + 1), buf.add(index), data.len - index - 1);
        data.len -= 1;
        removed
    }

    pub(crate) unsafe fn swap_remove(&self, data: &mut FSVecData, index: usize) -> T {
        if index >= data.len {
            panic!("Index out of bounds");
        }

        let buf = data.buf.as_ptr() as *mut T;
        let child = ptr::read(buf.add(index));
        ptr::copy(buf.add(data.len - 1), buf.add(index), 1);
        data.len -= 1;
        child
    }

    pub(crate) unsafe fn swap(&self, data: &mut FSVecData, index_1: usize, index_2: usize) {
        if index_1 >= data.len || index_2 >= data.len {
            panic!("Index out of bounds");
        }

        let buf = data.buf.as_ptr() as *mut T;
        ptr::swap(buf.add(index_1), buf.add(index_2));
    }

    pub(crate) unsafe fn at<'a>(&self, data: &'a FSVecData, index: usize) -> &'a T {
        if index >= data.len {
            panic!("Index out of bounds");
        }

        &*(data.buf.as_ptr() as *const T).add(index)
    }

    pub(crate) unsafe fn at_mut<'a>(&self, data: &'a mut FSVecData, index: usize) -> &'a mut T {
        if index >= data.len {
            panic!("Index out of bounds");
        }

        &mut *(data.buf.as_ptr() as *mut T).add(index)
    }

    pub(crate) unsafe fn wrap(self, vec: FSVecData) -> FSVec<T> {
        FSVec {
            data: vec,
            ops: self,
        }
    }
}

pub(crate) struct Iter<'a, T: 'a> {
    data: &'a FSVecData,
    index: usize,

    _phantom: PhantomData<T>,
}

impl<'a, T: 'a> Iter<'a, T> {
    pub(crate) unsafe fn new(data: &'a FSVecData) -> Iter<'a, T> {
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

pub(crate) struct FSVec<T> {
    data: FSVecData,
    ops: FSVecOps<T>,
}

impl<T> Drop for FSVec<T> {
    fn drop(&mut self) {
        unsafe {
            self.ops.drop(&mut self.data);
        }
    }
}

impl<T> Deref for FSVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { self.ops.as_slice(&self.data) }
    }
}

impl<T> DerefMut for FSVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.ops.as_slice_mut(&mut self.data) }
    }
}

impl<T> IntoIterator for FSVec<T> {
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
