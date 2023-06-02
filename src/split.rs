use std::{iter::repeat, ops::Sub};

use noisy_float::types::N64;

use crate::{
    bounds::{min_bounds, Bounded, Bounds},
    PREALLOCATE_CHILDREN,
};

/// Returns a pair of indices (a, b) where a < b. b is therefore also never zero.
fn worst_combination<N, const D: usize, Value>(children: &[Value]) -> (usize, usize)
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
    Value: Bounded<N, D>,
{
    if children.len() < 2 {
        panic!("Must have more than 2 children!");
    }
    (0..children.len() - 1)
        .flat_map(|lhs| repeat(lhs).zip((lhs + 1)..children.len()))
        .max_by_key(|(lhs, rhs)| {
            min_bounds(&children[*lhs].bounds(), &children[*rhs].bounds()).volume()
        })
        .unwrap()
}

/// Seeds splitting of values into two groups by finding two values which will
/// form the seeds of two groups. The seed of the first group is moved to
/// values[0], while the seed of the second group is returned.
fn seed_split_groups<N, const D: usize, Value>(values: &mut Vec<Value>) -> Value
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
    Value: Bounded<N, D>,
{
    let (i_1, i_2) = worst_combination(values);
    values.swap(0, i_1);
    values.remove(i_2)
}

fn best_candidate_for_group<N, const D: usize, Value>(
    children: &[Value],
    bounds: &Bounds<N, D>,
) -> Option<(usize, N64)>
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
    Value: Bounded<N, D>,
{
    children
        .into_iter()
        .enumerate()
        .map(|(i, value)| (i, min_bounds(&value.bounds(), bounds).volume()))
        .min_by_key(|(_, volume)| *volume)
}

/// Splits values into two groups. When it returns, values contains the values of
/// the first group while the other group is returned along with its minimum
/// bounds.
pub fn quadratic<N, const D: usize, Value>(
    max_children: usize,
    values: &mut Vec<Value>,
) -> (Bounds<N, D>, Bounds<N, D>, Vec<Value>)
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
    Value: Bounded<N, D>,
{
    if values.len() < 2 {
        panic!("Must have more than 2 children to split!");
    }

    let min_group_size = values.len() / 2;

    let mut group_2 = if PREALLOCATE_CHILDREN {
        let mut group_2 = Vec::with_capacity(max_children);
        group_2.push(seed_split_groups(values));
        group_2
    } else {
        vec![seed_split_groups(values)]
    };
    let (mut bounds_1, mut bounds_2) = (values[0].bounds(), group_2[0].bounds());

    let mut group_1_len = 1;
    // children is now partitioned such that children[0..group_1_len] is group_1
    // and children[group_1_len..] is the remaining children to be distributed
    // into groups.
    // When the loop terminates, children is group_1.
    while group_1_len < values.len() {
        let remaining = &values[group_1_len..];
        let (candidate_1, candidate_2) = (
            best_candidate_for_group(remaining, &bounds_1).unwrap(),
            best_candidate_for_group(remaining, &bounds_2).unwrap(),
        );

        let add_to_group_1 = if candidate_1.1 < candidate_2.1 {
            group_2.len() + remaining.len() - 1 >= min_group_size
        } else {
            group_1_len + remaining.len() - 1 == min_group_size
        };

        if add_to_group_1 {
            bounds_1 = min_bounds(&bounds_1, &remaining[candidate_1.0].bounds());
            values.swap(group_1_len + candidate_1.0, group_1_len);
            group_1_len += 1;
        } else {
            bounds_2 = min_bounds(&bounds_2, &remaining[candidate_2.0].bounds());
            group_2.push(values.remove(candidate_2.0))
        }
    }

    (bounds_1, bounds_2, group_2)
}
