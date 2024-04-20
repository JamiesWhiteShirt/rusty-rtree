use std::{ptr::NonNull, slice};

pub(crate) enum GetOnlyResult<T> {
    Multiple,
    Only(T),
    None,
}

/// Tries to get the only element in the iterator.
/// Returns `GetOnlyResult::Only` if there is only one element.
/// Returns `GetOnlyResult::Multiple` if there is more than one element.
/// Returns `GetOnlyResult::None` if there are no elements.
pub(crate) fn get_only<I: Iterator>(mut iter: I) -> GetOnlyResult<I::Item> {
    if let Some(first) = iter.next() {
        if iter.next().is_some() {
            GetOnlyResult::Multiple
        } else {
            GetOnlyResult::Only(first)
        }
    } else {
        GetOnlyResult::None
    }
}

pub(crate) fn empty_slice<'a, T>() -> &'a mut [T] {
    // SAFETY: For any type `T`, `NonNull::<T>::dangling().as_ptr()` is a valid
    // pointer to an array of length 0. Because the slice is empty, it is safe to
    // treat it as mutable and to assign it any lifetime.
    unsafe { slice::from_raw_parts_mut(NonNull::<T>::dangling().as_ptr(), 0) }
}
