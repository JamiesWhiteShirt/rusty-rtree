use std::{mem::swap, ops::Sub};

use noisy_float::types::N64;
use num_traits::Float;

use crate::bounds::{min_bounds, Bounded, Bounds};

/// Returns a pair of indices (a, b) where a < b. b is therefore also never zero.
fn worst_combination<N, const D: usize, Value>(
    values: &[Value],
    overflow_value: &Value,
) -> (usize, usize)
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
    Value: Bounded<N, D>,
{
    if values.len() < 1 {
        panic!("Must have more than 2 values!");
    }
    let mut greatest_volume_increase = N64::neg_infinity();
    let mut idx_1 = 0;
    let mut idx_2 = values.len();
    for i in 0..values.len() {
        let volume = min_bounds(&values[i].bounds(), &overflow_value.bounds()).volume();
        if volume > greatest_volume_increase {
            greatest_volume_increase = volume;
            idx_1 = i;
        }
    }

    for i in 0..values.len() - 1 {
        for j in i..values.len() {
            let volume = min_bounds(
                &values[i].bounds(),
                &min_bounds(&values[j].bounds(), &overflow_value.bounds()),
            )
            .volume();
            if volume > greatest_volume_increase {
                greatest_volume_increase = volume;
                idx_1 = i;
                idx_2 = j;
            }
        }
    }

    (idx_1, idx_2)
}

/// Seeds splitting of values into two groups by finding two values which will
/// form the seeds of two groups. The seed of the first group is moved to
/// values[0], while the seed of the second group is returned.
fn seed_split_groups<N, const D: usize, Value>(
    values: &mut Vec<Value>,
    mut overflow_value: Value,
) -> Value
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
    Value: Bounded<N, D>,
{
    let (i_1, i_2) = worst_combination(values, &overflow_value);
    values.swap(0, i_1);
    if i_2 < values.len() {
        swap(&mut values[i_2], &mut overflow_value)
    }
    overflow_value
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
    min_children: usize,
    values: &mut Vec<Value>,
    overflow_value: Value,
) -> (Bounds<N, D>, Bounds<N, D>, Vec<Value>)
where
    N: Ord + Clone + Sub<Output = N> + Into<f64>,
    Value: Bounded<N, D>,
{
    if values.len() < 1 {
        panic!("Must have more than 2 children to split!");
    }

    let mut group_2 = Vec::with_capacity(max_children);
    group_2.push(seed_split_groups(values, overflow_value));
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
            group_2.len() + remaining.len() - 1 >= min_children
        } else {
            group_1_len + remaining.len() - 1 == min_children
        };

        if add_to_group_1 {
            bounds_1 = min_bounds(&bounds_1, &remaining[candidate_1.0].bounds());
            values.swap(group_1_len + candidate_1.0, group_1_len);
            group_1_len += 1;
        } else {
            bounds_2 = min_bounds(&bounds_2, &remaining[candidate_2.0].bounds());
            group_2.push(values.remove(group_1_len + candidate_2.0))
        }
    }

    (bounds_1, bounds_2, group_2)
}
