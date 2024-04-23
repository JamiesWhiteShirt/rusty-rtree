use std::{
    alloc,
    borrow::Borrow,
    cell::{Cell, UnsafeCell},
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
};

#[repr(C)]
struct RcMutBox<T: ?Sized> {
    shared: Cell<bool>,
    value: UnsafeCell<T>,
}

pub(crate) struct RcMutAlloc<T: ?Sized>(NonNull<RcMutBox<T>>);

impl<T: ?Sized> RcMutAlloc<T> {
    unsafe fn allocate_for_layout(
        layout: alloc::Layout,
        mem_to_box: impl FnOnce(*mut u8) -> *mut RcMutBox<T>,
    ) -> *mut RcMutBox<T> {
        let layout = alloc::Layout::new::<RcMutBox<()>>()
            .extend(layout)
            .unwrap()
            .0
            .pad_to_align();

        match NonNull::new(alloc::alloc(layout)) {
            Some(ptr) => mem_to_box(ptr.as_ptr()),
            None => alloc::handle_alloc_error(layout),
        }
    }
}

impl<T: ?Sized> RcMutAlloc<T> {
    /// Creates a new `RcMutAlloc` for the given pointer to a value. The pointer does not need to be
    /// valid for reads or writes, but it must carry correct metadata for a value. The metadata will
    /// be used to determine the layout of the value when allocating the underlying box, and it will
    /// be used as the metadata for references to the value.
    ///
    /// # Safety
    ///
    /// The safety of this function is dependent on [alloc::Layout::for_value_raw].
    pub(crate) unsafe fn new_for_value_raw(t: *const T) -> Self {
        let box_ = Self::allocate_for_layout(alloc::Layout::for_value_raw(t), |mem| {
            // The metadata for Box<T> is the same as the metadata for T, but there is no way to
            // convince the compiler that <T as ptr::Pointee>::Metadata == <RcMutBox<T> as ptr::Pointee>::Metadata
            // The best we can do is cast *mut T to *mut RcMutBox<T>.
            ptr::from_raw_parts_mut::<T>(mem as *mut (), ptr::metadata(t)) as *mut _
                as *mut RcMutBox<T>
        });
        ptr::addr_of_mut!((*box_).shared).write(Cell::new(false));
        Self(NonNull::new(box_).unwrap())
    }

    fn clone(&self) -> Self {
        unsafe {
            let box_ = Self::allocate_for_layout(
                alloc::Layout::for_value(self.0.as_ref().value.borrow()),
                |mem| ptr::from_raw_parts_mut(mem as *mut (), ptr::metadata(self.0.as_ptr())),
            );
            ptr::addr_of_mut!((*box_).shared).write(Cell::new(false));
            Self(NonNull::new(box_).unwrap())
        }
    }

    /// Initializes the value in the box with the given callback and returns a mutable reference to
    /// it. The pointer passed to the function points to uninitialized memory that must be
    /// initialized during the callback.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the value is fully initialized when the callback returns.
    pub(crate) unsafe fn make_mut_init_raw(&mut self, init_value: impl FnOnce(*mut T)) -> RcMut<T> {
        unsafe {
            let shared = &*ptr::addr_of_mut!((*self.0.as_ptr()).shared);
            if shared.get() {
                *self = self.clone();
            }
            let shared = &*ptr::addr_of_mut!((*self.0.as_ptr()).shared);
            shared.set(true);
            let value = ptr::addr_of_mut!((*self.0.as_ptr()).value) as *mut T;
            init_value(value);
            RcMut(self.0)
        }
    }
}

impl<T> RcMutAlloc<[T]> {
    unsafe fn allocate_for_slice(len: usize) -> *mut RcMutBox<[T]> {
        let layout = alloc::Layout::array::<T>(len).unwrap();
        Self::allocate_for_layout(layout, |mem| {
            ptr::slice_from_raw_parts(mem.cast::<T>(), len) as *mut RcMutBox<[T]>
        })
    }

    pub(crate) fn new_slice(len: usize) -> Self {
        let box_ = unsafe { Self::allocate_for_slice(len) };
        unsafe {
            ptr::addr_of_mut!((*box_).shared).write(Cell::new(false));
        }
        Self(NonNull::new(box_).unwrap())
    }

    pub(crate) fn len(&self) -> usize {
        let value = unsafe { ptr::addr_of!(self.0.as_ref().value) } as *const [T];
        value.len()
    }

    pub(crate) fn make_mut_slice_from_fn(&mut self, mut cb: impl FnMut(usize) -> T) -> RcMut<[T]> {
        unsafe {
            self.make_mut_init_raw(|value_ptr| {
                let value = &mut *(value_ptr as *mut [MaybeUninit<T>]);
                for i in 0..value.len() {
                    value[i].write(cb(i));
                }
            })
        }
    }
}

impl<T> RcMutAlloc<T> {
    pub(crate) fn new() -> Self {
        let box_ = unsafe { Self::allocate_for_layout(alloc::Layout::new::<T>(), <*mut u8>::cast) };
        unsafe {
            ptr::addr_of_mut!((*box_).shared).write(Cell::new(false));
        }
        Self(NonNull::new(box_).unwrap())
    }

    /// Initializes the value in the box with the given value and returns a mutable reference to it.
    /// If there already is a mutable reference to the value, a new box is allocated instead.
    pub(crate) fn make_mut(&mut self, value: T) -> RcMut<T> {
        unsafe { self.make_mut_init_raw(|value_ptr| ptr::write(value_ptr, value)) }
    }
}

impl<T: ?Sized> Drop for RcMutAlloc<T> {
    fn drop(&mut self) {
        unsafe {
            // We must deallocate if it is not shared
            if !ptr::addr_of_mut!((*self.0.as_ptr()).shared)
                .as_ref()
                .unwrap()
                .replace(false)
            {
                // The value is uninitialized, so we don't need to drop it, but we still need to
                // deallocate the memory
                let layout = alloc::Layout::for_value(self.0.as_ref());
                alloc::dealloc(self.0.as_ptr().cast(), layout);
            }
        }
    }
}

/// A mutable reference to a value allocated in a `RcMutAlloc`.
pub struct RcMut<T: ?Sized>(NonNull<RcMutBox<T>>);

impl<T: ?Sized> Drop for RcMut<T> {
    fn drop(&mut self) {
        // We must deallocate if it is not shared
        if !unsafe { self.0.as_ref() }.shared.replace(false) {
            unsafe {
                ptr::drop_in_place(self.0.as_ptr());
                let layout = alloc::Layout::for_value(self.0.as_ref());
                alloc::dealloc(self.0.as_ptr().cast(), layout);
            }
        }
    }
}

impl<T: ?Sized> Deref for RcMut<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_ref().value.get().as_ref().unwrap_unchecked() }
    }
}

impl<T: ?Sized> DerefMut for RcMut<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.as_ref().value.get().as_mut().unwrap_unchecked() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reusable_box() {
        let mut box_ = RcMutAlloc::<u32>::new();
        let mut value0 = box_.make_mut(41);
        let value1 = box_.make_mut(42);
        assert_eq!(*value0, 41);
        assert_eq!(*value1, 42);
        *value0 = 43;
        assert_eq!(*value0, 43);
        assert_eq!(*value1, 42);
    }
}
