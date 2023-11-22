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
