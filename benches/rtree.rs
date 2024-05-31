use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rusty_rtree::{
    bounds::SAABB, filter::BoundedIntersectsFilter, vector::SVec, RTree, RTreeConfig,
};

fn insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");
    for max_children in [4, 8, 16, 32, 64, 128, 256] {
        let min_children = max_children / 4;
        let mut rng = StdRng::from_seed([
            0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD,
            0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF,
            0xDE, 0xAD, 0xBE, 0xEF,
        ]);

        let mut tree = RTree::<SAABB<i32, 2>, SAABB<i32, 2>, i32>::new(RTreeConfig {
            max_children,
            min_children,
        });
        for i in 0..10000 {
            let min = SVec([rng.gen_range(0..991), rng.gen_range(0..991)]);
            let max = min + SVec([rng.gen_range(1..11), rng.gen_range(1..11)]);
            tree.insert(SAABB { min, max }, i);
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(max_children),
            &max_children,
            |b, _| {
                b.iter_batched(
                    || tree.clone(),
                    |mut tree| {
                        let min = SVec([rng.gen_range(0..991), rng.gen_range(0..991)]);
                        let max = min + SVec([10, 10]);
                        tree.insert(SAABB { min, max }, 0)
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

fn query(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");
    for max_children in [4, 8, 16, 32, 64, 128, 256] {
        let mut rng = StdRng::from_seed([
            0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD,
            0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF,
            0xDE, 0xAD, 0xBE, 0xEF,
        ]);
        let mut tree = RTree::<SAABB<i32, 2>, SAABB<i32, 2>, i32>::new(RTreeConfig {
            max_children,
            min_children: max_children / 2,
        });
        for i in 0..10000 {
            let min = SVec([rng.gen_range(0..991), rng.gen_range(0..991)]);
            let max = min + SVec([rng.gen_range(1..11), rng.gen_range(1..11)]);
            tree.insert(SAABB { min, max }, i);
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(max_children),
            &max_children,
            |b, _| {
                b.iter(|| {
                    let min = SVec([rng.gen_range(0..991), rng.gen_range(0..991)]);
                    let max = min + SVec([10, 10]);
                    for entry in
                        tree.filter_iter(BoundedIntersectsFilter::new_bounded(SAABB { min, max }))
                    {
                        black_box(entry);
                    }
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, insert, query,);
criterion_main!(benches);
