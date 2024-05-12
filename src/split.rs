use std::mem::swap;

use noisy_float::types::N64;
use num_traits::Float;

use crate::{
    bounds::{Bounded, Bounds, Volume},
    fc_vec::{FCVecContainer, FCVecRefMut},
};

/// Returns a pair of indices (a, b) where a < b. b is therefore also never zero.
fn worst_combination<B, Value>(values: &[Value], overflow_value: &Value) -> (usize, usize)
where
    B: Bounds + Volume,
    Value: Bounded<B>,
{
    if values.len() < 1 {
        panic!("Must have more than 2 values!");
    }
    let mut greatest_volume_increase = N64::neg_infinity();
    let mut idx_1 = 0;
    let mut idx_2 = values.len();
    for i in 0..values.len() {
        let volume = Bounds::union(&values[i].bounds(), &overflow_value.bounds()).volume();
        if volume > greatest_volume_increase {
            greatest_volume_increase = volume;
            idx_1 = i;
        }
    }

    for i in 0..values.len() - 1 {
        for j in i + 1..values.len() {
            let volume = Bounds::union(
                &values[i].bounds(),
                &Bounds::union(&values[j].bounds(), &overflow_value.bounds()),
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
fn seed_split_groups<B, Value>(values: &mut [Value], mut overflow_value: Value) -> Value
where
    B: Bounds + Volume,
    Value: Bounded<B>,
{
    let (i_1, i_2) = worst_combination(values, &overflow_value);
    values.swap(0, i_1);
    if i_2 < values.len() {
        swap(&mut values[i_2], &mut overflow_value)
    }
    overflow_value
}

fn best_candidate_for_group<B, Value>(children: &[Value], bounds: &B) -> Option<(usize, N64)>
where
    B: Bounds + Volume,
    Value: Bounded<B>,
{
    children
        .into_iter()
        .enumerate()
        .map(|(i, value)| (i, Bounds::union(&value.bounds(), bounds).volume()))
        .min_by_key(|(_, volume)| *volume)
}

pub trait Splitter<B> {
    /// Splits values into two groups. When it returns, values contains the values of the first
    /// group while the other group is returned along with its minimum bounds.
    fn split<'a, Value>(
        &mut self,
        min_children: usize,
        values: FCVecRefMut<'a, Value>,
        overflow_value: Value,
    ) -> (B, B, FCVecContainer<Value>)
    where
        Value: Bounded<B>;
}

pub struct QuadraticSplitter;

impl<B> Splitter<B> for QuadraticSplitter
where
    B: Bounds + Volume,
{
    fn split<'a, Value>(
        &mut self,
        min_children: usize,
        mut values: FCVecRefMut<'a, Value>,
        overflow_value: Value,
    ) -> (B, B, FCVecContainer<Value>)
    where
        Value: Bounded<B>,
    {
        if values.len() < 1 {
            panic!("Must have more than 2 children to split!");
        }

        let mut group_2 = values.ops().new();
        group_2.push(seed_split_groups(&mut *values, overflow_value));
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
                bounds_1 = Bounds::union(&bounds_1, &remaining[candidate_1.0].bounds());
                values.swap(group_1_len + candidate_1.0, group_1_len);
                group_1_len += 1;
            } else {
                bounds_2 = Bounds::union(&bounds_2, &remaining[candidate_2.0].bounds());
                group_2.push(values.swap_remove(group_1_len + candidate_2.0))
            }
        }

        (bounds_1, bounds_2, group_2)
    }
}
