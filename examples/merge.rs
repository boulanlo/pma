use itertools::Itertools;
use pma::merge::Merge;
use pma::parallel_merge::ParallelMerge;
use rayon_adaptive::prelude::*;
use std::iter::repeat_with;
use time::precise_time_ns;

fn bench<I, J, S, O, T>(setup: S, op: O, test: T) -> u64
where
    S: Fn() -> I,
    O: Fn(I) -> J,
    T: Fn(J) -> bool,
{
    repeat_with(setup)
        .take(100)
        .map(|i| {
            let start = precise_time_ns();
            let j = op(i);
            let end = precise_time_ns();
            assert!(test(j));
            end - start
        })
        .sum::<u64>()
        / 100
}

fn random_vecs(size: usize) -> (Vec<u32>, Vec<u32>) {
    let (mut v1, mut v2): (Vec<u32>, Vec<u32>) =
        repeat_with(rand::random::<u32>).tuples().take(size).unzip();
    v1.sort();
    v2.sort();
    (v1, v2)
}

fn main() {
    println!(
        "seq merge: {}",
        bench(
            || random_vecs(1_000_000),
            |(v1, v2)| {
                let m = Merge {
                    slices: [v1.as_slice(), v2.as_slice()],
                    indexes: [0; 2],
                };
                let out: Vec<u32> = m.cloned().collect();
                out
            },
            |o| o.iter().tuples().all(|(a, b)| a <= b) && o.len() == 2_000_000
        )
    );
    println!(
        "par merge (in seq): {}",
        bench(
            || random_vecs(1_000_000),
            |(v1, v2)| {
                let m = ParallelMerge {
                    left: v1.into_par_iter(),
                    right: v2.into_par_iter(),
                }
                .to_sequential();
                let out: Vec<u32> = m.cloned().collect();
                out
            },
            |o| o.iter().tuples().all(|(a, b)| a <= b) && o.len() == 2_000_000
        )
    )
}
